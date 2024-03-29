/*******************************************/
//ROS HEADERS
/********************************************/
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <std_srvs/Trigger.h>
#include <sensor_msgs/Image.h>

/*********************************************/
//PCL HEADERS
/**********************************************/
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/filters/covariance_sampling.h>
#include <pcl/filters/normal_space.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/mls.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/crop_box.h>
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Core>

/***********************************************/
//STANDARD HEADERS
/************************************************/
#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sys/types.h>
#include <sys/wait.h>
/*************************************************/
//Other Libraries
/*************************************************/

#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/make_shared.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <chrono>
#include <unistd.h>

/************************************************/
//LOCAL HEADERS
/***********************************************/
#include "pcl_ros_utilities/pcl_ros_utilities.hpp"
#include "utilities/OccupancyGrid.hpp"

using namespace std;
using namespace pcl;
using namespace boost::interprocess;

constexpr double kResolution = 0.005;
constexpr double kZmin = 0.28;
constexpr double kZmax = 0.6;

/**
 * @brief The data collection class deals with collecting pointclouds, color and depth images.
 */

class PointcloudFusion
{
	public:
		PointcloudFusion(ros::NodeHandle& nh, const std::string& tsdf_frame,std::vector<double>& box,string directory_name);

	private:
        //Subscribers
		/** @brief Subscriber that listens to incoming point clouds */
		ros::Subscriber point_cloud_sub_;
		/** @brief Buffer used to store locations of the camera (*/
		tf2_ros::Buffer tf_buffer_;
		/** @brief Listener used to look up tranforms for the location of the camera */
		tf2_ros::TransformListener robot_tform_listener_;
        void onReceivedPointCloud(sensor_msgs::PointCloud2Ptr cloud_in);
        //Services
		ros::ServiceServer save_point_cloud_;
		ros::ServiceServer reset_service_;
		ros::ServiceServer start_service_;
		ros::ServiceServer stop_service_;
		ros::ServiceServer process_clouds_;
		bool reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool start(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool stop(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool getFusedCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
        void updateStates();
        void addPoints();
        void cleanGrid();
        //Objects
		std::string fusion_frame_;
		std::string pointcloud_frame_;
		string directory_name_;
        std::deque<std::pair<Eigen::Affine3d,sensor_msgs::PointCloud2Ptr>> clouds_;
        std::deque<std::tuple<Eigen::Affine3d,pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> clouds_processed_;
        OccupancyGrid grid_;
        vector<double>& bounding_box_;

        bool start_;
        bool cloud_subscription_started_;
        //Publishers
		ros::Publisher processed_cloud_;
        std::vector<std::thread> threads_;
        std::mutex mtx_;  
        std::mutex proc_mtx_;  
        std::mutex grid_mtx_;  
        std::condition_variable cv_;
};

PointcloudFusion::PointcloudFusion(ros::NodeHandle& nh,const std::string& fusion_frame,vector<double>& box,string directory_name)
	: robot_tform_listener_(tf_buffer_)
	, fusion_frame_(fusion_frame)
	, bounding_box_(box)
	, directory_name_(directory_name)
{  // Subscribe to point cloud
    point_cloud_sub_ = nh.subscribe("input_point_cloud", 100, &PointcloudFusion::onReceivedPointCloud,this);
	// point_cloud_sub_ = nh.subscribe("input_point_cloud", 1, &PointcloudFusion::onReceivedPointCloudDisplay,this);
	reset_service_= nh.advertiseService("reset",&PointcloudFusion::reset, this);
	start_service_= nh.advertiseService("start",&PointcloudFusion::start, this);
	stop_service_= nh.advertiseService("stop",&PointcloudFusion::stop, this);
    process_clouds_ = nh.advertiseService("process",&PointcloudFusion::getFusedCloud, this);
	processed_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("pcl_fusion_node/processed_cloud_normals",1);
    start_ = false;
    cloud_subscription_started_ = false;
    grid_.setResolution(kResolution,kResolution,kResolution);
    grid_.setDimensions(box[0],box[1],box[2],box[3],box[4],box[5]);
    grid_.setK(2);
    grid_.construct();
    std::cout<<"Construction done.."<<std::endl;
    threads_.push_back(std::thread(&PointcloudFusion::addPoints, this));
    threads_.push_back(std::thread(&PointcloudFusion::updateStates, this));
    threads_.push_back(std::thread(&PointcloudFusion::cleanGrid,this));
}
vector<int> splitRGBData(float rgb)
{
    uint32_t data = *reinterpret_cast<int*>(&rgb);
    vector<int> d;
    int a[3]={16,8,1};
    for(int i=0;i<3;i++)
    {
        d.push_back((data>>a[i]) & 0x0000ff);
    }
    return d;
}

pcl::PointCloud<pcl::PointXYZRGB> pointCloud2ToPclXYZRGBOMP(const pcl::PCLPointCloud2& p)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.points.resize(p.row_step/p.point_step);
    #pragma omp parallel for \ 
        default(none) \
            shared(cloud,p) \
            num_threads(2)
    for(int i=0;i<p.row_step;i+=p.point_step)
    {
        vector<float> t;
        for(int j=0;j<3;j++)
        {
            if(p.fields[j].count==0)
            {
                continue;
            }
            float x;
            memcpy(&x,&p.data[i+p.fields[j].offset],sizeof(float));
            t.push_back(x);
        }
        float rgb_data;
        memcpy(&rgb_data,&p.data[i+p.fields[3].offset],sizeof(float));
        vector<int> c = splitRGBData(rgb_data);    
        pcl::PointXYZRGB point;
        point.x = t[0];
        point.y = t[1];
        point.z = t[2];
        uint32_t rgb = (static_cast<uint32_t>(c[0]) << 16 |
                static_cast<uint32_t>(c[1]) << 8 | static_cast<uint32_t>(c[2]));
        point.rgb = *reinterpret_cast<float*>(&rgb);
        cloud.points[i/p.point_step] = point;  
    }
    return cloud;
}   

void PointcloudFusion::addPoints()
{
    int counter = 0;
    while(ros::ok())
    {
        bool received_data = false;
        std::pair<Eigen::Affine3d,sensor_msgs::PointCloud2Ptr> cloud_data;
        mtx_.lock();
        if(clouds_.size()!=0)
        {
            cloud_data = clouds_[0];
            clouds_.erase(clouds_.begin());
            clouds_.shrink_to_fit();
            received_data = true;
            // std::cout<<"Pointcloud "<<counter++<<" added.."<<std::endl;
        }
        mtx_.unlock();
        if(received_data==false)
        {
            // std::cout<<"Pointcloud "<<counter++<<" added.."<<std::endl;
            sleep(1);//TODO: Conditional Wait.
            continue;
        }
        pcl::PCLPointCloud2 pcl_pc2;
        Eigen::Affine3d fusion_frame_T_camera;
        auto cloud_in = cloud_data.second;
        fusion_frame_T_camera = cloud_data.first;
        pcl_conversions::toPCL(*cloud_in, pcl_pc2); 

        auto cloud = pointCloud2ToPclXYZRGBOMP(pcl_pc2);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clipped(new pcl::PointCloud<pcl::PointXYZRGB>);
        // boxFilter.setInputCloud(body);
        // boxFilter.filter(*bodyFiltered);
        for(auto point:cloud.points)
            if(point.z<kZmax&&point.z>kZmin)
            {
                cloud_clipped->points.push_back(point);
            }
        // usleep(200000);
        counter++;
        proc_mtx_.lock();
        clouds_processed_.push_back(make_tuple(fusion_frame_T_camera,cloud_clipped));
        proc_mtx_.unlock();

    }
}

void PointcloudFusion::updateStates()
{
    int counter = 0;
    while(ros::ok())
    {
        std::tuple<Eigen::Affine3d,pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_data;
        bool received_data = false;
        proc_mtx_.lock();
        if(clouds_processed_.size()!=0)
        {
            cloud_data = clouds_processed_[0];
            clouds_processed_.pop_front(); 
            received_data = true;
        }
        proc_mtx_.unlock();
        if(received_data==false)
        {
            // std::cout<<"Pointcloud "<<counter<<" states updated.."<<std::endl;
            sleep(1);
            continue;
        }
        auto fusion_frame_T_camera = get<0>(cloud_data);
        auto cloud = get<1>(cloud_data);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud (*cloud, *cloud_transformed, fusion_frame_T_camera);
        Vector3f vp = {fusion_frame_T_camera(0,3),fusion_frame_T_camera(1,3),fusion_frame_T_camera(2,3)};
        grid_mtx_.lock();
        if(start_)
            grid_.addPoints<6>(cloud_transformed,vp);
        else
            grid_.addPoints<8>(cloud_transformed,vp);
        grid_mtx_.unlock();
        std::cout<<"Pointcloud "<<counter++<<" states updated.."<<std::endl;
    }
}

void PointcloudFusion::cleanGrid()
{
    while(ros::ok())
    {
        grid_mtx_.lock();
        if(grid_.state_changed)
        {
            if(start_==true)
            {
                std::cout<<"Started Cleaning.."<<std::endl;
                grid_.updateThicknessVectors<6,3>();
                std::cout<<"Finished Cleaning.."<<std::endl;
            }
            else
            {
                std::cout<<"Started Cleaning OMP.."<<std::endl;
                grid_.updateThicknessVectors<8,3>();
                std::cout<<"Finished Cleaning OMP.."<<std::endl;
            }
        }
        grid_mtx_.unlock();
        std::cout<<"Grid Cleaned.."<<std::endl;
        sleep(5);    
    }
}

void PointcloudFusion::onReceivedPointCloud(sensor_msgs::PointCloud2Ptr cloud_in)
{
    pointcloud_frame_ = cloud_in->header.frame_id;
    cloud_subscription_started_ = true;
    if(start_)
    {
        Eigen::Affine3d fusion_frame_T_camera = Eigen::Affine3d::Identity();
        try
        {
            geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_.lookupTransform(fusion_frame_, cloud_in->header.frame_id,ros::Time(0));
            // std::cout<<transform_fusion_frame_T_camera.header.stamp<<" --- "<<cloud_in->header.stamp<<" : "<<ros::Time::now()<<std::endl;
            fusion_frame_T_camera = tf2::transformToEigen(transform_fusion_frame_T_camera); 
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN("%s", ex.what());
            return;
        }
        mtx_.lock();
        clouds_.push_back(make_pair(fusion_frame_T_camera,cloud_in));
        mtx_.unlock();
    }
}

bool PointcloudFusion::reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"RESET"<<std::endl;
    start_ = false;
    res.success=true;
    clouds_.clear();
    cloud_subscription_started_ = false;
	return true;
}

bool PointcloudFusion::start(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
	std::cout<<"START"<<std::endl;
    start_ = true;
	res.success=true;
	return true;
}

bool PointcloudFusion::stop(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
	std::cout<<"STOP"<<std::endl;
    start_ = false;
	res.success=true;
	return true;
}

bool PointcloudFusion::getFusedCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    bool merging_finished = false;
    while(ros::ok()&&merging_finished==false)//TODO: Busy waiting use conditional wait.
    {
        mtx_.lock();
        if(clouds_.size()==0)
            merging_finished = true;
        mtx_.unlock();
        if(merging_finished==true)
        {
            proc_mtx_.lock();
            if(clouds_processed_.size()!=0)
                merging_finished=false;
            proc_mtx_.unlock();
        }
        sleep(1);
    }
    std::string cloud_location = directory_name_ + "/test_cloud.pcd";
    std::string meta_location = directory_name_ + "/meta.csv";
    std::cout<<"Downloading cloud and metadata to "<<cloud_location<<" and "<<meta_location<<std::endl;
    grid_.downloadData(cloud_location,meta_location);
#if 0
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_classified(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_50(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_100(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_150(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_200(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_250(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_300(new pcl::PointCloud<pcl::PointXYZRGB>);
    grid_mtx_.lock();
    grid_.downloadHQ(cloud);
    grid_.downloadHQ(cloud_50,50);
    grid_.downloadHQ(cloud_100,100);
    grid_.downloadHQ(cloud_150,150);
    grid_.downloadHQ(cloud_200,200);
    grid_.downloadHQ(cloud_250,250);
    grid_.downloadHQ(cloud_300,300);
    grid_.download(cloud_normals);
    grid_.downloadClassified(cloud_classified);
    grid_mtx_.unlock();
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud_classified->height = 1;
    cloud_classified->width = cloud_classified->points.size();
    cloud_normals->height = 1;
    cloud_normals->width = cloud_normals->points.size();

    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud.pcd",*cloud);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_50.pcd",*cloud_50);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_100.pcd",*cloud_100);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_150.pcd",*cloud_150);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_200.pcd",*cloud_200);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_250.pcd",*cloud_250);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_300.pcd",*cloud_300);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_classified.pcd",*cloud_classified);
    pcl::io::savePCDFileASCII (directory_name_+"/test_cloud_normals.pcd",*cloud_normals);
#endif
    grid_.clearVoxels();
    return true;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "fusion_node");
	ros::NodeHandle pnh("~");
	string fusion_frame="";
	pnh.param<std::string>("fusion_frame", fusion_frame, "fusion_frame");
	string directory_name = "";
	pnh.param<std::string>("directory_name", directory_name, "./");
	std::vector<double> bounding_box;
	pnh.param("bounding_box", bounding_box, std::vector<double>());
	PointcloudFusion dc(pnh,fusion_frame,bounding_box,directory_name); 
	ros::Rate loop_rate(31);
	while(ros::ok())
	{
		ros::spinOnce();
		loop_rate.sleep();
	}   
	return 0;
}
