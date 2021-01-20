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

Eigen::MatrixXd rot2eul(Eigen::Matrix3d rot_mat, std::string seq)
{
    int rot_idx[3];
    for (int i=0; i<3; ++i)
    {
        if(seq[i]=='X' || seq[i]=='x')
            rot_idx[i] = 0;
        else if(seq[i]=='Y' || seq[i]=='y')
            rot_idx[i] = 1;
        else if(seq[i]=='Z' || seq[i]=='z')
            rot_idx[i] = 2;
    }   
    Eigen::MatrixXd eul_angles(1,3);
    Eigen::Vector3d eul_angles_vec;
    eul_angles_vec = rot_mat.eulerAngles(rot_idx[0], rot_idx[1], rot_idx[2]);
    eul_angles(0,0) = eul_angles_vec[0];
    eul_angles(0,1) = eul_angles_vec[1];
    eul_angles(0,2) = eul_angles_vec[2];
    return eul_angles;
}

vector<double> affineMatrixToVector(Eigen::Affine3d transformation)
{
    Eigen::MatrixXd rotation(3,3);
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            rotation(i,j) = transformation(i,j);
    auto eul_angles = rot2eul(rotation,"ZYX");
    return {transformation(0,3),transformation(1,3),transformation(2,3),eul_angles(0,0),eul_angles(0,1),eul_angles(0,2)};
}

/**
 * @brief The data collection class deals with collecting pointclouds, color and depth images.
 */

class PointcloudFusion
{
    public:
        PointcloudFusion(ros::NodeHandle& nh, const std::string& tsdf_frame,std::vector<double>& box,string directory_name,std::string flange_frame);

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
        ros::ServiceServer capture_service_,finish_service_;
        bool capture(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
        bool finish(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res);
        //Objects
        std::string fusion_frame_;
        std::string flange_frame_;
        std::string pointcloud_frame_;
        string directory_name_;
        vector<double>& bounding_box_;
        ofstream ofile_;

        bool capture_;
        bool cloud_subscription_started_;
};

PointcloudFusion::PointcloudFusion(ros::NodeHandle& nh,const std::string& fusion_frame,vector<double>& box,string directory_name,string flange_frame)
    : robot_tform_listener_(tf_buffer_)
    , fusion_frame_(fusion_frame)
    , bounding_box_(box)
    , directory_name_(directory_name)
{  // Subscribe to point cloud
    point_cloud_sub_ = nh.subscribe("input_point_cloud", 100, &PointcloudFusion::onReceivedPointCloud,this);
    // point_cloud_sub_ = nh.subscribe("input_point_cloud", 1, &PointcloudFusion::onReceivedPointCloudDisplay,this);
    capture_service_= nh.advertiseService("capture",&PointcloudFusion::capture, this);
    finish_service_= nh.advertiseService("finish",&PointcloudFusion::finish, this);
    capture_ = false;
    cloud_subscription_started_ = false;
    directory_name_ = directory_name;
    flange_frame_ = flange_frame;
    ofile_.open (directory_name_+"/BaseToFlange.txt");
    std::cout<<"Initialization done.."<<std::endl;
}

void PointcloudFusion::onReceivedPointCloud(sensor_msgs::PointCloud2Ptr cloud_in)
{
    static int counter = 0;
    pointcloud_frame_ = cloud_in->header.frame_id;
    cloud_subscription_started_ = true;
    if(capture_)
    {
        Eigen::Affine3d fusion_frame_T_camera = Eigen::Affine3d::Identity();
        try
        {
            geometry_msgs::TransformStamped transform_fusion_frame_T_camera = tf_buffer_.lookupTransform(fusion_frame_, "tool0",ros::Time(0));
            // std::cout<<transform_fusion_frame_T_camera.header.stamp<<" --- "<<cloud_in->header.stamp<<" : "<<ros::Time::now()<<std::endl;
            fusion_frame_T_camera = tf2::transformToEigen(transform_fusion_frame_T_camera); 
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN("%s", ex.what());
            return;
        }
        //TODO: Save the pointcloud and the flange transformation in the required format.
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_in, pcl_pc2); 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZRGB>);
        auto cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        for(auto point:cloud.points)
            if(point.z<1.0)
                cloud_temp->points.push_back(point);
        std::cout<<"The transformation:"<<std::endl;
        auto _6dof = affineMatrixToVector(fusion_frame_T_camera);
        for(int i=0;i<6;i++)
            std::cout<<_6dof[i]<<" ";
        ofile_<<_6dof[0]*1000.0<<","<<_6dof[1]*1000.0<<","<<_6dof[2]*1000.0;
        for(int i=3;i<6;i++)
            ofile_<<","<<_6dof[i];
        ofile_<<"\n";
        std::cout<<std::endl;
        std::cout<<"Saving the cloud.."<<counter+1<<std::endl;
        cloud_temp->height = 1;
        cloud_temp->width = cloud_temp->points.size();
        pcl::io::savePCDFileASCII (directory_name_+"/pos_"+to_string(++counter)+".pcd",*cloud_temp);
        capture_ = false;
    }
}

bool PointcloudFusion::capture(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"Capture"<<std::endl;
    capture_ = true;
    res.success=true;
    return true;
}

bool PointcloudFusion::finish(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
    std::cout<<"Finishing"<<std::endl;
    ofile_.close();
    capture_ = false;
    res.success=true;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fusion_node");
    ros::NodeHandle pnh("~");
    string fusion_frame="";
    string flange_frame="";
    pnh.param<std::string>("fusion_frame", fusion_frame, "fusion_frame");
    pnh.param<std::string>("flange_frame", flange_frame, "flange_frame");
    string directory_name = "";
    pnh.param<std::string>("directory_name", directory_name, "./");
    std::vector<double> bounding_box;
    pnh.param("bounding_box", bounding_box, std::vector<double>());
    PointcloudFusion dc(pnh,fusion_frame,bounding_box,directory_name,flange_frame); 
    ros::Rate loop_rate(11);
    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }   
    return 0;
}
