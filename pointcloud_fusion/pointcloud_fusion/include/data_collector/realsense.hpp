#ifndef REALSENSE_HPP
#define REALSENSE_HPP

/*******************************************/
//ROS HEADERS
/********************************************/
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/PolygonMesh.h>
#include <std_srvs/Trigger.h>
#include <sensor_msgs/Image.h>


/*********************************************/
//PCL HEADERS
/**********************************************/
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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

#include <Eigen/Dense>
#include <Eigen/Core>
#include <cv.h>
#include <highgui.h>

/***********************************************/
//STANDARD HEADERS
/************************************************/
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

namespace filter
{
	/**
	* @brief The filter class deals with fusing the point clouds into one.
	*/
	class PclFilter
	{
	public:
		PclFilter(ros::NodeHandle& nh,vector<string> &topics);
	    Mat color_image,depth_image,combined_depth_image;
	    Eigen::MatrixXd combined_depth_image_matrix,useful_pixels,consistent_pixels;
	    bool color_done,depth_done,camera_done;
	    // ros::NodeHandle nh;
	    Eigen::VectorXd K;
	    double fx,cx,fy,cy;
	    std::string frame_id;
	    pcl::PointCloud<pcl::PointXYZRGB> makePointCloud();

	private:
		void onReceivedRawPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_in);

	    void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr& camMsg);

		void colorImageCb(const sensor_msgs::ImageConstPtr& msg);

		void depthImageCb(const sensor_msgs::ImageConstPtr& msg);



		/** @brief Subscriber that listens to incoming point clouds */
		ros::Subscriber point_cloud_sub_;

        ros::Subscriber image_sub_, depth_sub_,info_sub_;

		ros::Publisher publish_cloud;

		int frame_count;

		pcl::PointCloud<pcl::PointXYZRGB> combined_point_cloud;

		ros::ServiceServer capture_point_cloud_srv;

		bool capture;

		bool onCapturePointCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
	};
}  // namespace filter

#endif