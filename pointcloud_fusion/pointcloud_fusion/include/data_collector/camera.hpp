#ifndef CAMERA_HPP
#define CAMERA_HPP

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
#include <sensor_msgs/image_encodings.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>



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

using namespace cv;

class Camera
{
public:
    Mat color_image,depth_image;
    bool color_done,depth_done,camera_done;
    ros::NodeHandle nh;
    ros::Subscriber image_sub_, depth_sub_,info_sub_;
    Eigen::VectorXd K;
    double fx,cx,fy,cy;
    string frame_id;
    Camera(const vector<string>& topics);//The order of topics: color,depth,camera_info
    void colorImageCb(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCb(const sensor_msgs::ImageConstPtr& msg);
    void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr& camMsg);
    pcl::PointCloud<pcl::PointXYZRGB> makePointCloud();
};

Camera::Camera(const vector<string>& topics)
{
    color_done=false;
    depth_done=false;
    camera_done=false;
    K.resize(9);
    K=Eigen::VectorXd::Zero(9);
    info_sub_ = nh.subscribe(topics[0],1,&Camera::cameraInfoCb,this);
    if(topics.size()>1)
    depth_sub_ = nh.subscribe(topics[1], 1,&Camera::depthImageCb,this);
    if(topics.size()>2)
    image_sub_ = nh.subscribe(topics[2], 1,&Camera::colorImageCb,this);
}

void Camera::colorImageCb(const sensor_msgs::ImageConstPtr& msg)
{
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
    color_image=cv_ptr->image.clone();
    color_done=true;
}

void Camera::depthImageCb(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    depth_image=cv_ptr->image.clone();
    depth_done=true;
}

void Camera::cameraInfoCb(const sensor_msgs::CameraInfoConstPtr& msg)
{
	if(msg->K.size()==0) return;
    for(int i=0;i<9;i++)
        K(i)=msg->K.at(i);
    fx=K[0],cx=K[2],fy=K[4],cy=K[5];
    camera_done=true;
    frame_id=msg->header.frame_id;
    if (msg != nullptr)
    {
        info_sub_.shutdown();
    }
}

pcl::PointCloud<pcl::PointXYZRGB> Camera::makePointCloud()
{
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    Size s = depth_image.size();
    int width = s.width;
    int height = s.height;
    pcl_cloud.width = width*height;
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = true;
    pcl_cloud.header.frame_id = frame_id;
    int i=0;
    for(int r=0;r<width;r++)
    { 
        for(int c=0;c<height;c++)
        {    
            if(depth_image.at<unsigned short>(r,c)==0||depth_image.at<unsigned short>(r,c)!=depth_image.at<unsigned short>(r,c)) continue;
            pcl_cloud.points[i].r = color_image.at<Vec3b>(r,c)[2];
            pcl_cloud.points[i].g = color_image.at<Vec3b>(r,c)[1];
            pcl_cloud.points[i].b = color_image.at<Vec3b>(r,c)[0];
            pcl_cloud.points[i].z = depth_image.at<unsigned short>(r,c) * 0.001;
            pcl_cloud.points[i].x = pcl_cloud.points[i].z * ( (double)c - cx ) / (fx);
            pcl_cloud.points[i].y = pcl_cloud.points[i].z * ((double)r - cy ) / (fy);
            i++;
        }
    }
    pcl_cloud.width=i;
    pcl_cloud.resize(i);
    return pcl_cloud;
}
//To Do: Averaging, Point cloud generation...

#endif