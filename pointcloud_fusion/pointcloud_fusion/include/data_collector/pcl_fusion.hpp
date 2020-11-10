#ifndef PCL_FUSION_HPP
#define PCL_FUSION_HPP

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

/***********************************************/
//STANDARD HEADERS
/************************************************/
#include <iostream>
#include <cmath>

namespace fusion
{
	/**
	* @brief The fusion class deals with fusing the point clouds into one.
	*/
	class PclFusion
	{
	public:
		PclFusion(ros::NodeHandle& nh, const std::string& tsdf_frame,std::vector<double>& box);

	private:
		void onReceivedPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
		void onReceivedPointCloud2(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
		void pairAlign (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_src, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tgt, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output, Eigen::Matrix4f &final_transform, bool downsample);
		/** @brief Subscriber that listens to incoming point clouds */
		ros::Subscriber point_cloud_sub_;

    	ros::Subscriber point_cloud_sub2_;

		/** @brief Buffer used to store locations of the camera (*/
		tf2_ros::Buffer tf_buffer_;
		/** @brief Listener used to look up tranforms for the location of the camera */
		tf2_ros::TransformListener robot_tform_listener_;
		/** @brief Service that is called to trigger mesh generation and saving the mesh as an stl...TODO*/
		ros::ServiceServer generate_mesh_service_;
		/** @brief Service that resets the point clouds fused...TODO */
		ros::ServiceServer reset_fusion_service_;
		/** @brief Service that saves the point cloud */
		ros::ServiceServer save_point_cloud;
		/** @brief Used to track if the camera has moved. Only add image if it has */

		ros::ServiceServer capture_point_cloud;

		Eigen::Affine3d fusion_frame_T_camera_prev_;
		/** @brief TF frame associated with the TSDF volume. */
		std::string fusion_frame_;

		pcl::PointCloud<pcl::PointXYZRGB> combined_pcl;

		ros::Publisher publish_cloud;

		std::vector<double> bounding_box;

		ros::Subscriber saving_message_sub;

		bool savePointCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);

		bool resetFusion(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);

		bool capturePointCloud(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);

	};
}  // namespace fusion

#endif