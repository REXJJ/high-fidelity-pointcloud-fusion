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
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
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
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/************************************************/
//LOCAL HEADERS
/***********************************************/
#include "point_cloud_utilities/pcl_utilities.hpp"

using namespace std;
using namespace pcl;
using namespace cv;

namespace rex_utils
{
	void saveTransformation(string file_name,Eigen::Affine3d& transformation)
	{
		ofstream f(file_name);
		for(int i=0;i<transformation.rows();i++)
		{
			f<<transformation(i,0);
			for(int j=1;j<transformation.cols();j++)
				f<<","<<transformation(i,j);
			f<<"\n";
		}
	}

	void print(){std::cout<<std::endl;}
	template<typename T,typename... Args>
		void print(T Contents, Args... args) 
		{
#ifndef NDEBUG
			std::cout<< (Contents) <<" ";print(args...);
#endif
		}
}

using namespace rex_utils;
/**
 * @brief The data collection class deals with collecting pointclouds, color and depth images.
 */
class DataCollection
{
	public:
		DataCollection(ros::NodeHandle& nh, const std::string& tsdf_frame,std::vector<double>& box,string directory_name);

	private:
		/** @brief Subscriber that listens to incoming point clouds */
		ros::Subscriber point_cloud_sub_;
		ros::Subscriber color_image_sub_;
		ros::Subscriber depth_image_sub_;
		ros::Subscriber camera_info_sub_;
		/** @brief Buffer used to store locations of the camera (*/
		tf2_ros::Buffer tf_buffer_;
		/** @brief Listener used to look up tranforms for the location of the camera */
		tf2_ros::TransformListener robot_tform_listener_;
		/** @brief Service that is called to trigger mesh generation and saving the mesh as an stl...TODO*/
		tf2_ros::Buffer tf_buffer_depth_;
		/** @brief Listener used to look up tranforms for the location of the camera */
		tf2_ros::TransformListener robot_tform_listener_depth_;
		ros::ServiceServer save_point_cloud_;
		/** @brief Used to track if the camera has moved. Only add image if it has */
		ros::ServiceServer capture_point_cloud_;
		ros::ServiceServer reset_service_;
		std::string fusion_frame_;
		pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_;
		ros::Publisher publish_cloud_;
		std::vector<double> bounding_box_;
		ros::Subscriber saving_message_sub_;
		bool collect_data_;
		string directory_name_;
		ros::Publisher data_collected_;
		void onReceivedPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
		bool captureData(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		bool reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
		int take_,point_cloud_no_,depth_image_no_;
		bool point_cloud_saved_,color_image_saved_,depth_image_saved_;
		void resetCounters();
		bool allSaved();
		void sendSavedMessage();
		void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr& msg);
		void colorImageCb(const sensor_msgs::ImageConstPtr& msg);
		void depthImageCb(const sensor_msgs::ImageConstPtr& msg);
		int max_frames_;
		string depth_image_frame_;
};

DataCollection::DataCollection(ros::NodeHandle& nh,const std::string& fusion_frame,vector<double>& box,string directory_name)
	: robot_tform_listener_(tf_buffer_)
	, robot_tform_listener_depth_(tf_buffer_depth_)
	, fusion_frame_(fusion_frame)
	, bounding_box_(box)
	, directory_name_(directory_name)
{  // Subscribe to point cloud
	point_cloud_sub_ = nh.subscribe("input_point_cloud", 1, &DataCollection::onReceivedPointCloud,this);
	color_image_sub_ = nh.subscribe("input_color_image", 1, &DataCollection::colorImageCb,this);
	depth_image_sub_ = nh.subscribe("input_depth_image", 1, &DataCollection::depthImageCb,this);
	camera_info_sub_ = nh.subscribe("input_camera_information", 1, &DataCollection::cameraInfoCb,this);
	reset_service_=nh.advertiseService("reset",&DataCollection::reset, this);
	capture_point_cloud_ = nh.advertiseService("capture_data",&DataCollection::captureData,this);
	data_collected_ = nh.advertise<std_msgs::String>("data_collected",1);
	take_=0;
	point_cloud_no_=0,depth_image_no_=0;
	max_frames_=1;
	point_cloud_saved_=false,color_image_saved_=false,depth_image_saved_=false;
}

void DataCollection::cameraInfoCb(const sensor_msgs::CameraInfoConstPtr& msg)
{
	std::cout<<"In Camera Info Call Back"<<std::endl;
	if(msg==nullptr) return;
	VectorXd K=Eigen::VectorXd::Zero(9);
	for(int i=0;i<9;i++)
		K(i)=msg->K.at(i);
	ofstream f(directory_name_+"/transforms/camera.csv");
	f<<K(0);
	for(int i=1;i<9;i++)
		f<<","<<K(i);
	f.close();
	camera_info_sub_.shutdown();
}

void DataCollection::colorImageCb(const sensor_msgs::ImageConstPtr& msg)
{
	if(collect_data_==false||color_image_saved_==true)
		return;
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	Mat color_image=cv_ptr->image.clone();
	string file_name=directory_name_+"/color_images/image"+to_string(take_)+".jpg";
	imwrite(file_name,color_image);
	color_image_saved_=true;
	print("Color Image Saved...");
	if(allSaved())
		sendSavedMessage();
}

void DataCollection::depthImageCb(const sensor_msgs::ImageConstPtr& msg)
{
	auto frame_id=msg->header.frame_id;
	if(collect_data_==false||depth_image_saved_==true)
		return;
	cv_bridge::CvImagePtr cv_ptr;
	print("Depth Frame: ",frame_id);
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	Eigen::Affine3d fusion_frame_T_camera = Eigen::Affine3d::Identity();
	try
	{
		geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_depth_.lookupTransform(fusion_frame_, frame_id,ros::Time(0));
		fusion_frame_T_camera = tf2::transformToEigen(transform_fusion_frame_T_camera); 
	}
	catch (tf2::TransformException& ex)
	{
		ROS_WARN("%s", ex.what());
		return;
	}
	Mat  depth_image=cv_ptr->image.clone();
	string file_name=directory_name_+"/depth_image/image"+to_string(take_)+to_string(depth_image_no_++)+".pcd";
	imwrite(file_name,depth_image);
	print("Depth Image Saved...");
	if(depth_image_no_==max_frames_)
	{
		string output_transformation=directory_name_+"/transforms/depth_image_transforms"+to_string(take_)+to_string(depth_image_no_)+".txt";
		saveTransformation(output_transformation,fusion_frame_T_camera);	
		//New Changes...
		depth_image_saved_=true;
		if(allSaved())
			sendSavedMessage();

	}
}

void DataCollection::onReceivedPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
{
	// Convert to useful point cloud format
	auto frame_id=cloud_in->header.frame_id;
    cout<<frame_id<<endl;
	if(collect_data_==false)
		return;
	if(point_cloud_saved_)
		return;
	pcl::PCLPointCloud2 pcl_pc2;
	pcl_conversions::toPCL(*cloud_in, pcl_pc2);
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
	print("Inside Pointcloud Function");
	Eigen::Affine3d fusion_frame_T_camera = Eigen::Affine3d::Identity();
	try
	{
		geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_.lookupTransform(fusion_frame_, cloud_in->header.frame_id,ros::Time(0));
		fusion_frame_T_camera = tf2::transformToEigen(transform_fusion_frame_T_camera); 
	}
	catch (tf2::TransformException& ex)
	{
		ROS_WARN("%s", ex.what());
		return;
	}
	string output_pcd=directory_name_+"/pointclouds/cloud"+to_string(take_)+to_string(point_cloud_no_++)+".pcd";
	pcl::io::savePCDFileASCII (output_pcd,cloud);
	if(point_cloud_no_==max_frames_)
	{
		string output_transformation=directory_name_+"/transforms/pointcloud_transforms"+to_string(take_)+".txt";
		saveTransformation(output_transformation,fusion_frame_T_camera);	
		//New Changes...
		point_cloud_saved_=true;
		if(allSaved())
			sendSavedMessage();

	}
}

void DataCollection::sendSavedMessage()
{
	std_msgs::String msg;
	std::stringstream ss;
	ss << "Data saved... ";
	msg.data = ss.str();
	data_collected_.publish(msg);
	take_++;
	resetCounters();
}

bool DataCollection::reset(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
	std::cout<<"RESET"<<std::endl;
	res.success=true;
	collect_data_ = false;
	take_=0;
	return true;
}

bool DataCollection::captureData(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
	std::cout<<"CAPTURE"<<std::endl;
	res.success = true;
	collect_data_ = true;
	return true;
}

void DataCollection::resetCounters()
{
	point_cloud_no_=0,depth_image_no_=0;
	collect_data_ = false;
	point_cloud_saved_=false,color_image_saved_=false,depth_image_saved_=false;
}

bool DataCollection::allSaved()
{
	return point_cloud_saved_&&color_image_saved_&&depth_image_saved_;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "fusion_node");
	ros::NodeHandle pnh("~");
	string fusion_frame="";
	pnh.param<std::string>("fusion_frame", fusion_frame, "fusion_frame");
	string directory_name = "/home/rex/REX_WS/Test_WS/POINT_CLOUD_STITCHING/data/data_08-31/";
	std::vector<double> bounding_box;
	pnh.param("bounding_box", bounding_box, std::vector<double>());
	DataCollection dc(pnh,fusion_frame,bounding_box,directory_name); 
	ros::Rate loop_rate(1);
	while(ros::ok())
	{
		ros::spinOnce();
		loop_rate.sleep();
	}   
	return 0;
}
