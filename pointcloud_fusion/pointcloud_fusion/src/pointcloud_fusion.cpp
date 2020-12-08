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
#include "utilities/Volume.hpp"

using namespace std;
using namespace pcl;
using namespace boost::interprocess;


constexpr double ball_radius = 0.01;
constexpr double cylinder_radius = 0.001;
constexpr double downsample_radius = 0.005;

Eigen::Vector3f getNormal(const pcl::PointCloud<pcl::PointXYZ> cloud)
{
  Eigen::MatrixXd lhs (cloud.size(), 3);
  Eigen::VectorXd rhs (cloud.size());
  for (size_t i = 0; i < cloud.size(); ++i)
  {
    const auto& pt = cloud.points[i];
    lhs(i, 0) = pt.x;
    lhs(i, 1) = pt.y;
    lhs(i, 2) = 1.0;

    rhs(i) = -1.0 * pt.z;
  }
  Eigen::Vector3d params = lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
  Eigen::Vector3d normal (params(0), params(1), 1.0);
  auto length = normal.norm();
  normal /= length;
  params(2) /= length;
  return {normal(0), normal(1), normal(2)};
}

Vector3f projectPointToVector(Vector3f pt, Vector3f norm_pt, Vector3f n)
{
    Vector3f d_xyz = n*ball_radius;
    Vector3f a = norm_pt - d_xyz;
    Vector3f b = norm_pt + d_xyz;
    Vector3f ap = (a-pt);
    Vector3f ab = (a-b);
    Vector3f p = a - (ap.dot(ab)/ab.dot(ab))*ab;
    return p;
}

void process(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr centroids,pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bw (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::copyPointCloud(*cloud,*cloud_bw);

    std::cout<<"XYZ pointcloud created."<<std::endl;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_temp (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_bw);

    std::cout<<"Tree created."<<std::endl;
   
    unsigned long long int good_centers = 0;
    for(int i=0;i<centroids->points.size();i++)
    {
        if(i%100==0)
        {
            std::cout<<i<<" out of "<<centroids->points.size()<<" done."<<std::endl;
        }
        auto pt = centroids->points[i];
        // pcl::PointXYZ ptxyz = pcl::PointXYZ({pt.x,pt.y,pt.z});
        pcl::PointXYZ ptxyz;
        ptxyz.x = pt.x;
        ptxyz.y = pt.y;
        ptxyz.z = pt.z;
        vector<int> indices;
        vector<float> dist;
        auto t=tree->radiusSearch(ptxyz,ball_radius,indices,dist,0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr points( new pcl::PointCloud<pcl::PointXYZ> );
        for(auto ids:indices)
        {
            PointXYZ pt_temp;
            pt_temp.x = cloud->points[ids].x;
            pt_temp.y = cloud->points[ids].y;
            pt_temp.z = cloud->points[ids].z;
            points->points.push_back(pt_temp);
        }

        auto points_downsampled = PCLUtilities::downsample<pcl::PointXYZ>(points,0.001);

        int good_points = 0;
        if(indices.size()>10)
        {
            Vector3f normal = getNormal(points_downsampled);
            Vector3f norm_pt;
            norm_pt<<pt.x,pt.y,pt.z;
            Vector3f pt_pro;
            pt_pro<<0,0,0;
            double weights = 0.0;
            for(auto ids:indices)
            {
                PointXYZ pt_temp;
                pt_temp.x = cloud->points[ids].x;
                pt_temp.y = cloud->points[ids].y;
                pt_temp.z = cloud->points[ids].z;
                Vector3f pt_loc;
                pt_loc<<pt_temp.x,pt_temp.y,pt_temp.z;
                Vector3f projected_points = projectPointToVector(pt_loc, norm_pt,normal);
                double distance_to_normal = (pt_loc - projected_points).norm();
                // std::cout<<"Distance To Normal: "<<distance_to_normal<<" Distance to normal Pt: "<<(pt_loc-norm_pt).norm()<<std::endl;
                if(distance_to_normal<cylinder_radius)
                {
                    pt_pro+= projected_points;
                    weights += (1.0);
                    // cout<<"Points: "<<pt_loc<<" Projected Points: "<<projected_points<<" Normal: "<<normal<<endl;
                    good_points++;

                }
            }
            if(good_points==0)
            {
                cout<<"No Good Points.."<<endl;
                continue;
            }
            pt_pro/=weights;
            PointXYZRGB pt_processed;
            pt_processed.x = pt_pro(0);
            pt_processed.y = pt_pro(1);
            pt_processed.z = pt_pro(2);
            pt_processed.r = pt.r;
            pt_processed.g = pt.g;
            pt_processed.b = pt.b;
            good_centers++;
            processed->points.push_back(pt_processed); 
        }
    }
    processed->height = 1;
    processed->width = good_centers;
}

/**
 * @brief The data collection class deals with collecting pointclouds, color and depth images.
 */

struct shm_remove
{
    shm_remove() { shared_memory_object::remove("shared_memory"); }
    ~shm_remove(){ shared_memory_object::remove("shared_memory"); }
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
		void onReceivedPointCloudDisplay(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
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
        void fuseAndFilter();
        void estimateNormalsAndFuse();
        //Objects
		std::string fusion_frame_;
		std::string pointcloud_frame_;
		string directory_name_;
        std::deque<std::pair<Eigen::Affine3d,sensor_msgs::PointCloud2Ptr>> clouds_;
        pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_pcl_ptr_;
        pcl::PointCloud<pcl::PointXYZRGB> combined_pcl_display_;
        pcl::PointCloud<pcl::PointXYZ> combined_pcl_bw_;
		pcl::PointCloud<pcl::Normal> combined_pcl_normals_;
		std::vector<double> bounding_box_;
        bool start_;
        bool cloud_subscription_started_;
        bool display_;
        //Publishers
		ros::Publisher processed_cloud_;
        std::vector<std::thread> threads_;
        std::mutex mtx_;           
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
	processed_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("pcl_fusion_node/processed_cloud",1);
    start_ = false;
    cloud_subscription_started_ = false;
    display_ = false;
    combined_pcl_ptr_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    threads_.push_back(std::thread(&PointcloudFusion::estimateNormalsAndFuse, this));
}

void PointcloudFusion::estimateNormalsAndFuse()
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
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
        auto cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        auto processed_cloud = PCLUtilities::downsample<pcl::PointXYZRGB>(cloud,0.01);
        for(auto point:processed_cloud.points)
            if(point.z<2.0&&point.z>0.28)
            {
                pcl::PointXYZ pt;
                pt.x = point.x;
                pt.y = point.y;
                pt.z = point.z;
                cloud_temp->points.push_back(pt);
            }
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_temp);
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> normalEstimator;
        normalEstimator.setInputCloud(cloud_temp);
        normalEstimator.setSearchMethod(tree);
        normalEstimator.setRadiusSearch(0.01);
        normalEstimator.setViewPoint(0,0,0);
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
        normalEstimator.compute(*cloud_normals);

        pcl::PointCloud<pcl::PointNormal> cloud_transformed;
        pcl::transformPointCloudWithNormals(*cloud_normals,cloud_transformed, fusion_frame_T_camera);
        std::cout<<"Pointcloud processed."<<std::endl;
        std::cout<<counter++<<std::endl;
    }
}

void PointcloudFusion::fuseAndFilter()
{
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
        pcl::PointCloud<pcl::PointXYZRGB> cloud_temp;
        auto cloud = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
        for(auto point:cloud.points)
            if(point.z<2.0&&point.z>0.28)
                cloud_temp.points.push_back(point);
        pcl::PointCloud<pcl::PointXYZRGB> cloud_transformed;
        pcl::transformPointCloud (cloud_temp, cloud_transformed, fusion_frame_T_camera);
        for(auto point:cloud_transformed.points)
            // if(point.z>0.0&&point.x>0.94&&point.x<1.33&&point.y>-0.38&&point.y<0.38)//TODO: Remove hardcoded values.
            if(point.z>0.0)//TODO: Remove hardcoded values.
                combined_pcl_ptr_->points.push_back(point);        
        std::cout<<"Pointcloud received."<<std::endl;
        std::cout<<cloud_in->header<<std::endl;
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
        // static int counter = 0;
        // std::cout<<counter++<<std::endl;
    }
}

void PointcloudFusion::onReceivedPointCloudDisplay(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
{
    // if(display_==false)
    //     return;
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
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*cloud_in, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_temp;
    cloud_temp = PCLUtilities::pointCloud2ToPclXYZRGB(pcl_pc2);
    for(auto point:cloud_temp.points)
        if(point.z<0.80)
            cloud.points.push_back(point);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_downsampled = PCLUtilities::downsample<pcl::PointXYZRGB>(cloud,0.008);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_transformed;
    pcl::transformPointCloud (cloud_downsampled, cloud_transformed, fusion_frame_T_camera);
    combined_pcl_display_.header.frame_id = fusion_frame_;
    combined_pcl_display_+=cloud_transformed;
    PCLUtilities::publishPointCloud<pcl::PointXYZRGB>(combined_pcl_display_,processed_cloud_);  
    std::cout<<"Fusion Frame: "<<fusion_frame_<<std::endl;
    std::cout<<"Size of the cloud: "<<combined_pcl_display_.points.size()<<std::endl;
    std::cout<<"Published the combined cloud."<<std::endl; 
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
        sleep(1);
    }
    std::cout<<combined_pcl_ptr_->points.size()<<std::endl;
    auto processed_cloud = PCLUtilities::downsample<pcl::PointXYZRGB>(combined_pcl_ptr_,0.005);
    std::cout<<combined_pcl_ptr_->points.size()<<std::endl;
    std::cout<<processed_cloud.points.size()<<std::endl;
    // PCLUtilities::publishPointCloud<pcl::PointXYZRGB>(processed_cloud,processed_cloud_);
    combined_pcl_ptr_->height = 1;
    combined_pcl_ptr_->width = combined_pcl_ptr_->points.size();
    processed_cloud.height = 1;
    processed_cloud.width = processed_cloud.points.size();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed_new(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout<<"Going to process the clouds."<<std::endl;
#if 0
    process(combined_pcl_ptr_,processed_cloud.makeShared(),processed_new);
    processed_new->height = 1;
    processed_new->width = processed_new->points.size();
    std::cout<<"Processed Cloud: "<<processed_new->points.size()<<std::endl;
#else
    MatrixXf m = combined_pcl_ptr_->getMatrixXfMap().transpose();
    std::cout<<m.rows()<<" "<<m.cols()<<std::endl;
    unsigned long long int cloud_size = m.rows()*m.cols();
    std::cout<<cloud_size<<std::endl;
    shared_memory_object shm_obj
        (create_only                  //only create
         ,"shared_memory"              //name
         ,read_write                   //read-write mode
        );
    shm_obj.truncate(cloud_size*sizeof(float));
    mapped_region region(shm_obj, read_write);
    memcpy(region.get_address(),m.data(),region.get_size());
    std::cout<<"Memory Mapping Done."<<std::endl;
#endif

    // pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test.pcd",*combined_pcl_ptr_);
    // pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_downsampled.pcd",processed_cloud);
    // pcl::io::savePCDFileASCII ("/home/rflin/Desktop/test_filtered.pcd",*processed_new);
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
