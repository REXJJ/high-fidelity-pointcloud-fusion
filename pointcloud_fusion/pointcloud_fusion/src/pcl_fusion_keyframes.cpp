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
		bool capture(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool getFusedCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_combined_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_combined_filtered_;
        //Objects
		std::string fusion_frame_;
		std::string pointcloud_frame_;
		string directory_name_;
        vector<double>& bounding_box_;

        bool start_;
        bool cloud_subscription_started_;
        //Publishers
		ros::Publisher processed_cloud_;
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
	start_service_= nh.advertiseService("capture",&PointcloudFusion::capture, this);
    process_clouds_ = nh.advertiseService("process",&PointcloudFusion::getFusedCloud, this);
	processed_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("pcl_fusion_node/processed_cloud_normals",1);
    start_ = false;
    cloud_subscription_started_ = false;
    cloud_combined_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_combined_filtered_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout<<"Construction done.."<<std::endl;
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

        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_in, pcl_pc2); 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        auto cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        for(auto point:cloud.points)
            if(point.z<2.0)
                cloud_temp->points.push_back(point);
        pcl::transformPointCloud (*cloud_temp, *cloud_transformed, fusion_frame_T_camera);
        // PCLUtilities::downsample_ptr<pcl::PointXYZRGB>(cloud_transformed,cloud_transformed,0.005);
        for(auto point:cloud_transformed->points)
            cloud_combined_->points.push_back(point);
        cloud_temp.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud_transformed.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        for(auto point:cloud.points)
            if(point.z<0.6&&point.z>0.3)
                cloud_temp->points.push_back(point);
        pcl::transformPointCloud (*cloud_temp, *cloud_transformed, fusion_frame_T_camera);
        for(auto point:cloud_transformed->points)
            cloud_combined_filtered_->points.push_back(point);
    start_ = false;
    }
}

bool PointcloudFusion::reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"RESET"<<std::endl;
    start_ = false;
    res.success=true;
    cloud_subscription_started_ = false;
	return true;
}

bool PointcloudFusion::capture(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
	std::cout<<"START"<<std::endl;
    start_ = true;
	res.success=true;
	return true;
}

bool PointcloudFusion::getFusedCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"Downloading cloud."<<std::endl;
    PCLUtilities::downsample_ptr<pcl::PointXYZRGB>(cloud_combined_,cloud_combined_,0.005);
    cloud_combined_->height = 1;
    cloud_combined_->width = cloud_combined_->points.size();
    PCLUtilities::downsample_ptr<pcl::PointXYZRGB>(cloud_combined_filtered_,cloud_combined_filtered_,0.005);
    cloud_combined_filtered_->height = 1;
    cloud_combined_filtered_->width = cloud_combined_filtered_->points.size();
    pcl::io::savePCDFileASCII (directory_name_+"/keyframe_cloud_downsampled.pcd",*cloud_combined_);
    pcl::io::savePCDFileASCII (directory_name_+"/keyframe_cloud_filtered_downsampled.pcd",*cloud_combined_filtered_);
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
