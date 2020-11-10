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

/*************************************************/
//Other Libraries
/*************************************************/

#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/make_shared.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
/************************************************/
//LOCAL HEADERS
/***********************************************/
#include "pcl_ros_utilities/pcl_ros_utilities.hpp"

using namespace std;
using namespace pcl;

/**
 * @brief The data collection class deals with collecting pointclouds, color and depth images.
 */
class PointcloudFusion
{
	public:
		PointcloudFusion(ros::NodeHandle& nh, const std::string& tsdf_frame,std::vector<double>& box,string directory_name);
        void captureTransformations();

	private:
        //Subscribers
		/** @brief Subscriber that listens to incoming point clouds */
		ros::Subscriber point_cloud_sub_;
		/** @brief Buffer used to store locations of the camera (*/
		tf2_ros::Buffer tf_buffer_;
		/** @brief Listener used to look up tranforms for the location of the camera */
		tf2_ros::TransformListener robot_tform_listener_;
		void onReceivedPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
        //Services
		ros::ServiceServer save_point_cloud_;
		ros::ServiceServer reset_service_;
		ros::ServiceServer start_service_;
		ros::ServiceServer stop_service_;
		ros::ServiceServer process_clouds_;
		bool reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool start(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool stop(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool filterAndFuse(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
        //Objects
		std::string fusion_frame_;
		std::string pointcloud_frame_;
		string directory_name_;
        std::vector<std::pair<ros::Time,pcl::PointCloud<pcl::PointXYZRGB>>> clouds_;
		pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_;
		std::vector<double> bounding_box_;
        bool start_;
        bool cloud_subscription_started_;
        std::vector<std::pair<ros::Time,Eigen::Affine3d>> transformations_;
        //Publishers
		ros::Publisher processed_cloud_;
};

PointcloudFusion::PointcloudFusion(ros::NodeHandle& nh,const std::string& fusion_frame,vector<double>& box,string directory_name)
	: robot_tform_listener_(tf_buffer_)
	, fusion_frame_(fusion_frame)
	, bounding_box_(box)
	, directory_name_(directory_name)
{  // Subscribe to point cloud
	point_cloud_sub_ = nh.subscribe("input_point_cloud", 1, &PointcloudFusion::onReceivedPointCloud,this);
	reset_service_= nh.advertiseService("reset",&PointcloudFusion::reset, this);
	start_service_= nh.advertiseService("start",&PointcloudFusion::start, this);
	stop_service_= nh.advertiseService("stop",&PointcloudFusion::stop, this);
    process_clouds_ = nh.advertiseService("process",&PointcloudFusion::filterAndFuse, this);
	processed_cloud_ = nh.advertise<std_msgs::String>("pcl_fusion_node/processed_cloud",1);
    start_ = false;
    cloud_subscription_started_ = false;
    // pointcloud_frame_ = "map";
}

int counter = 0;

void PointcloudFusion::onReceivedPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
{
    pointcloud_frame_ = cloud_in->header.frame_id;
    cloud_subscription_started_ = true;
    if(start_)
    {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_in, pcl_pc2);
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        if(counter++%10==0)
            clouds_.push_back(make_pair(cloud_in->header.stamp,cloud));
        std::cout<<"Pointcloud received."<<std::endl;
        std::cout<<cloud_in->header<<std::endl;
    }
}

bool PointcloudFusion::reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"RESET"<<std::endl;
    start_ = false;
    res.success=true;
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

bool PointcloudFusion::filterAndFuse(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    typedef pair<ros::Time, Eigen::Affine3d> trans;
    for(int i=0;i<clouds_.size();i++)
    {
        auto cloud = clouds_[i].second;
        auto query = make_pair(clouds_[i].first,Eigen::Affine3d::Identity());
        auto result = std::lower_bound(transformations_.begin(),transformations_.end(),query,[](trans a,trans b)->bool { return a.first < b.first; });
        auto transformation = result->second;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_transformed;
        pcl::transformPointCloud (cloud, cloud_transformed, transformation);
        std::cout<<cloud_transformed.points.size()<<std::endl;
        combined_pcl_ = combined_pcl_ + cloud_transformed;
        std::cout<<"Query: "<<query.first<<" Result: "<<result->first<<std::endl;
        std::cout<<"Processing cloud number: "<<i<<" out of "<<clouds_.size()<<std::endl;
    }
    std::cout<<combined_pcl_.points.size()<<std::endl;
    auto processed_cloud = PCLUtilities::downsample<pcl::PointXYZRGB>(combined_pcl_,0.01);
    std::cout<<combined_pcl_.points.size()<<std::endl;
    // PCLUtilities::publishPointCloud<pcl::PointXYZRGB>(combined_pcl_,processed_cloud_);
    pcl::io::savePCDFileASCII ("/home/rex/REX_WS/Test_WS/test.pcd",processed_cloud );
    std::cout<<"Fusion Done..."<<std::endl;
    res.success=true;
    return true;
}

void PointcloudFusion::captureTransformations()
{
    if(cloud_subscription_started_==false)
        return;
    Eigen::Affine3d fusion_frame_T_camera = Eigen::Affine3d::Identity();
    bool found = false;
    try
    {
        geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_.lookupTransform(fusion_frame_, pointcloud_frame_,ros::Time(0));
        // geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_.lookupTransform(fusion_frame_, cloud_in->header.frame_id,cloud_in->header.stamp);
        fusion_frame_T_camera = tf2::transformToEigen(transform_fusion_frame_T_camera); 
        if(start_)
            transformations_.push_back(make_pair(transform_fusion_frame_T_camera.header.stamp,fusion_frame_T_camera));
        found = true;
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN("%s", ex.what());
        return;
    }
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
	ros::Rate loop_rate(30);
	while(ros::ok())
	{
        dc.captureTransformations();//TODO: Find a way to abstract this out of the main loop.
		ros::spinOnce();
		loop_rate.sleep();
	}   
	return 0;
}
