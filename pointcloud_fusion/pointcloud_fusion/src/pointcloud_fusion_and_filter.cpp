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

/*************************************************/
//Other Libraries
/*************************************************/

#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/make_shared.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

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
        void onReceivedPointCloud(const sensor_msgs::PointCloud2Ptr& cloud_in);
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
        void estimateNormals();
        void updateStates();
        //Objects
		std::string fusion_frame_;
		std::string pointcloud_frame_;
		string directory_name_;
        std::deque<std::pair<Eigen::Affine3d,sensor_msgs::PointCloud2Ptr>> clouds_;
        std::deque<std::tuple<Eigen::Affine3d,pcl::PointCloud<pcl::PointXYZRGB>::Ptr,pcl::PointCloud<pcl::PointNormal>::Ptr>> clouds_processed_;
        pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_pcl_ptr_;
        pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_display_;
        pcl::PointCloud<pcl::PointXYZ> combined_pcl_bw_;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr combined_pcl_normals_;
		std::vector<double> bounding_box_;
        OccupancyGrid grid_;

        bool start_;
        bool cloud_subscription_started_;
        bool display_;
        //Publishers
		ros::Publisher processed_cloud_;
        std::vector<std::thread> threads_;
        std::mutex mtx_;  
        std::mutex proc_mtx_;    
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
    grid_.setDimensions(0.7,1.7,0.2,1.2,0,1);
    grid_.construct();
    grid_.setK(2);
    std::cout<<"Construction done.."<<std::endl;
    combined_pcl_ptr_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    combined_pcl_normals_.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    threads_.push_back(std::thread(&PointcloudFusion::estimateNormals, this));
    threads_.push_back(std::thread(&PointcloudFusion::updateStates, this));
}

void PointcloudFusion::estimateNormals()
{
    int counter = 0;
    while(ros::ok())
    {
        std::pair<Eigen::Affine3d,sensor_msgs::PointCloud2Ptr> cloud_data;
        bool received_data = false;
        mtx_.lock();
        if(clouds_.size()!=0)
        {
            cloud_data = clouds_[0];
            clouds_.pop_front(); 
            received_data = true;
        }
        mtx_.unlock();
        if(received_data==false)
        {
            sleep(1);//TODO: Conditional Wait.
            continue;
        }
        pcl::PCLPointCloud2 pcl_pc2;
        auto cloud_in = cloud_data.second;
        auto fusion_frame_T_camera = cloud_data.first;
        pcl_conversions::toPCL(*cloud_in, pcl_pc2); 
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bw (new pcl::PointCloud<pcl::PointXYZ>);
        auto cloud_input = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for(auto point:cloud_input.points)
        {
            if(point.z<0.81&&point.z>0.28)//TODO: Change hardcoded values.
            {
                pcl::PointXYZ pt;
                pt.x = point.x;
                pt.y = point.y;
                pt.z = point.z;
                cloud_bw->points.push_back(pt);
                cloud->points.push_back(point);
            }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr normals_root(new pcl::PointCloud<pcl::PointXYZ>);
        PCLUtilities::downsample_ptr<pcl::PointXYZ>(cloud_bw,normals_root,0.02);
        if(normals_root->points.size()>10000)
        {
            std::cout<<"Bad thing...."<<std::endl;
            continue;
        }
        std::cout<<"Downsampled and filtered points size : "<<normals_root->points.size()<<std::endl;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(normals_root);//TODO: Might need to change.
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator;
        normalEstimator.setInputCloud(normals_root);
        normalEstimator.setSearchMethod(tree);
        normalEstimator.setRadiusSearch(0.1);
        normalEstimator.setViewPoint(0,0,0);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        normalEstimator.compute(*cloud_normals);

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_output(new pcl::PointCloud<pcl::PointNormal>);
        for(int i=0;i<cloud_normals->points.size();i++)
        {
            pcl::PointNormal pt;
            pcl::PointXYZ ptxyz = normals_root->points[i];
            pcl::Normal ptn = cloud_normals->points[i];
            pt.x = ptxyz.x;
            pt.y = ptxyz.y;
            pt.z = ptxyz.z;
            pt.normal[0] = ptn.normal[0];
            pt.normal[1] = ptn.normal[1];
            pt.normal[2] = ptn.normal[2];
            cloud_normals_output->points.push_back(pt);
        }
        proc_mtx_.lock();
        clouds_processed_.push_back(make_tuple(fusion_frame_T_camera,cloud,cloud_normals_output));
        proc_mtx_.unlock();
        // pcl::PointCloud<pcl::PointNormal>::Ptr normals_transformed(new pcl::PointCloud<pcl::PointNormal>);
        // pcl::transformPointCloudWithNormals(*cloud_normals_output,*normals_transformed, fusion_frame_T_camera);      
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        // pcl::transformPointCloud (*cloud, *cloud_transformed, fusion_frame_T_camera);
        // grid_.updateStates(cloud_transformed,normals_transformed);
        std::cout<<"Pointcloud "<<counter++<<" normals calculated.."<<std::endl;
    }
}

void PointcloudFusion::updateStates()
{
    int counter = 0;
    while(ros::ok())
    {
        std::tuple<Eigen::Affine3d,pcl::PointCloud<pcl::PointXYZRGB>::Ptr,pcl::PointCloud<pcl::PointNormal>::Ptr> cloud_data;
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
            sleep(1);
            continue;
        }
        std::cout<<"In states updation thread."<<std::endl;
        auto fusion_frame_T_camera = get<0>(cloud_data);
        auto cloud = get<1>(cloud_data);
        auto normals = get<2>(cloud_data);
        pcl::PointCloud<pcl::PointNormal>::Ptr normals_transformed(new pcl::PointCloud<pcl::PointNormal>);
        pcl::transformPointCloudWithNormals(*normals,*normals_transformed, fusion_frame_T_camera);      
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud (*cloud, *cloud_transformed, fusion_frame_T_camera);
        grid_.updateStates(cloud_transformed,normals_transformed);
        std::cout<<"Pointcloud "<<counter++<<" states updated.."<<std::endl;
    }
}

void PointcloudFusion::onReceivedPointCloud(const sensor_msgs::PointCloud2Ptr& cloud_in)
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
