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
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
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

/**
 * @brief The data collection class deals with collecting pointclouds, color and depth images.
 */

struct shm_remove
{
    shm_remove() { shared_memory_object::remove("shared_memory_normals"); }
    ~shm_remove(){ shared_memory_object::remove("shared_memory_normals"); }
} remover;

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
        pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_pcl_ptr_;
        pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_display_;
        pcl::PointCloud<pcl::PointXYZ> combined_pcl_bw_;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr combined_pcl_normals_;
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_output_dbg;
		std::vector<double> bounding_box_;
        OccupancyGrid grid_;
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator_;

        pcl::CropBox<pcl::PointXYZRGB> box_filter_;

        bool start_;
        bool cloud_subscription_started_;
        bool display_;
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
    display_ = false;
    grid_.setResolution(0.005,0.005,0.005);
    grid_.setDimensions(0.80,1.80,-0.4,1.4,0,1);
    grid_.construct();
    grid_.setK(2);
    std::cout<<"Construction done.."<<std::endl;
    box_filter_.setMin(Eigen::Vector4f(-10, -10, 0.28, 1.0));
    box_filter_.setMax(Eigen::Vector4f(10, 10, 0.81, 1.0));
    normalEstimator_.setNumberOfThreads(8);
    combined_pcl_ptr_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    combined_pcl_normals_.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    cloud_normals_output_dbg.reset(new pcl::PointCloud<pcl::PointNormal>);
    threads_.push_back(std::thread(&PointcloudFusion::addPoints, this));
    threads_.push_back(std::thread(&PointcloudFusion::updateStates, this));
    threads_.push_back(std::thread(&PointcloudFusion::cleanGrid,this));
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
            received_data = true;
            std::cout<<"Pointcloud "<<counter++<<" added.."<<std::endl;
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

        auto cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clipped(new pcl::PointCloud<pcl::PointXYZRGB>);
        // boxFilter.setInputCloud(body);
        // boxFilter.filter(*bodyFiltered);
        for(auto point:cloud.points)
            if(point.z<2.0&&point.z>0.28)
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
        grid_mtx_.lock();
        grid_.addPoints(cloud_transformed);
        grid_mtx_.unlock();
        counter++;
        std::cout<<"Pointcloud "<<counter++<<" states updated.."<<std::endl;
    }
}

void PointcloudFusion::cleanGrid()
{
    int counter = 0;
    while(ros::ok())
    {
        grid_mtx_.lock();
        if(grid_.state_changed)
            grid_.updateStates();
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
        // Eigen::Affine3d fusion_frame_T_camera = Eigen::Affine3d::Identity();
        // try
        // {
        //     geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_.lookupTransform(fusion_frame_, cloud_in->header.frame_id,ros::Time(0));
        //     // std::cout<<transform_fusion_frame_T_camera.header.stamp<<" --- "<<cloud_in->header.stamp<<" : "<<ros::Time::now()<<std::endl;
        //     fusion_frame_T_camera = tf2::transformToEigen(transform_fusion_frame_T_camera); 
        // }
        // catch (tf2::TransformException& ex)
        // {
        //     ROS_WARN("%s", ex.what());
        //     return;
        // }
        mtx_.lock();
        clouds_.push_back(make_pair(Eigen::Affine3d::Identity(),cloud_in));
        mtx_.unlock();
    }
}

bool PointcloudFusion::reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"RESET"<<std::endl;
    start_ = false;
    res.success=true;

    clouds_.clear();
    combined_pcl_.clear();
    combined_pcl_ptr_->clear();
    combined_pcl_display_.clear();

    cloud_subscription_started_ = false;
    display_ = false;
    combined_pcl_ptr_.reset();
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
    std::cout<<"Downloading cloud."<<std::endl;
    grid_.downloadHQCloud(combined_pcl_normals_);
    pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_normals_combined.pcd",*combined_pcl_normals_);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_output(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    grid_.downloadCloud(cloud_output);
    pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_normals_combined_full.pcd",*cloud_output);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_output_reorganized(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    grid_.downloadReorganizedCloud(cloud_output_reorganized);
    pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_normals_combined_full_reorganized.pcd",*cloud_output_reorganized);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_output_reorganized_clean(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    grid_.downloadReorganizedCloud(cloud_output_reorganized_clean,true);
    pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_normals_combined_full_reorganized_clean.pcd",*cloud_output_reorganized_clean);
    // cloud_normals_output_dbg->height = 1;
    // cloud_normals_output_dbg->width = cloud_normals_output_dbg->points.size();
    // pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_normals_combined_full_raw_downsampled.pcd",*cloud_normals_output_dbg);
    std::cout<<"Fusion Done..."<<std::endl;
    res.success=true;
    return true;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "fusion_node");
	ros::NodeHandle pnh("~");
	string fusion_frame="";
	pnh.param<std::string>("fusion_frame", fusion_frame, "fusion_frame");
	string directory_name = "/home/rex/REX_WS/Catkin_WS/data/";
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
