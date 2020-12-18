#pragma once

/***********************************************/
//STANDARD HEADERS
/************************************************/
#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <chrono>
#include <unordered_map> 
#include <queue>
#include <fstream>
#include <thread>
#include <ctime>

/*********************************************/
//OTHER HEADERS
/**********************************************/
#include <Eigen/Dense>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/common/transforms.h>
#include <omp.h>

using namespace std;
using namespace pcl;
using namespace Eigen;

constexpr int kGoodPointsThreshold = 500;
constexpr double kBballRadius = 0.015;
constexpr double kCylinderRadius = 0.001;

struct Voxel
{
    bool occupied;
    void* data;
    Voxel()
    {
        occupied = false;
        data = nullptr;
    }
};

struct VoxelInfo
{
    Vector3f centroid;
    Vector3f normal;
    vector<Vector3f> buffer;
    bool normal_found;
    VoxelInfo()
    {
        normal_found = false;
    }
};

class OccupancyGrid
{
    public:
    double xmin_,xmax_,ymin_,ymax_,zmin_,zmax_;
    double xres_,yres_,zres_;
    int xdim_,ydim_,zdim_;
    int k_;
    vector<vector<vector<Voxel>>> voxels_;
    int counter;
    OccupancyGrid(){k_=2;xdim_=ydim_=zdim_=0;counter=0;};//TODO: Get k_ at compile time.
    void setDimensions(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax);
    void setResolution(float x,float y,float z);
    bool construct();
    tuple<int,int,int> getVoxelCoords(Vector3f point);
    bool validPoints(Vector3f point);
    bool addPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    bool download(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    Vector3f getVoxelCenter(int x,int y,int z)
    {
        Vector3f centroid = {xmin_+xres_*(x)+xres_/2.0,ymin_+yres_*(y)+yres_/2.0,zmin_+zres_*(z)+zres_/2.0};
        return centroid;
    }
};


bool OccupancyGrid::addPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    if(cloud==nullptr)
        return false;
    // #pragma omp parallel for \ 
    //     default(none) \
    //         shared(cloud) \
    //         num_threads(8)
    std::cout<<"Entered.."<<std::endl;
    std::cout<<counter++<<std::endl;
    for(int p=0;p<cloud->points.size();p++)
    {
        // std::cout<<p<<std::endl;
        auto pt = cloud->points[p];
        Vector3f point = {pt.x,pt.y,pt.z};
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(point);
        // std::cout<<point(0)<<" "<<point(1)<<" "<<point(2)<<std::endl;
        // std::cout<<x<<" "<<y<<" "<<z<<std::endl;
        if(validPoints(point)==false)
            continue;
        Voxel& voxel = voxels_[x][y][z];
        Vector3f ptv = {pt.x,pt.y,pt.z};
        if(voxel.occupied==true)
        {
            assert(voxel.data!=nullptr);
            VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxel.data);
            assert(data->buffer.size()>0);
            data->buffer.push_back(ptv);
        }
        else
        {
            voxel.occupied = true;
            VoxelInfo* data = new VoxelInfo();
            data->buffer.push_back(ptv);
            voxels_[x][y][z].data = reinterpret_cast<void*>(data);
            //TODO: Allocate 1000 before hand.
        }
    }
    std::cout<<"Exited.."<<std::endl;
    return true;
}

bool OccupancyGrid::download(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    if(cloud==nullptr)
        return false;
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                if(voxels_[x][y][z].occupied)
                {
                    VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxels_[x][y][z].data);
                    std::cout<<data->buffer.size()<<std::endl;
                    // assert(data->buffer.size()>0);
                    pcl::PointXYZRGB pt;
                    auto point = getVoxelCenter(x,y,z);
                    pt.x = point(0);
                    pt.y = point(1);
                    pt.z = point(2);
                    cloud->points.push_back(pt);
                }
    std::cout<<"Points: "<<cloud->points.size()<<std::endl;
}


void OccupancyGrid::setDimensions(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax)
{
    xmin_=xmin;
    xmax_=xmax;
    ymin_=ymin;
    ymax_=ymax;
    zmin_=zmin;
    zmax_=zmax;
}

void OccupancyGrid::setResolution(float x,float y,float z)
{
    xres_ = x;
    yres_ = y;
    zres_ = z;
}

bool OccupancyGrid::construct()
{
    xdim_ = (xmax_-xmin_)/xres_;
    ydim_ = (ymax_-ymin_)/yres_;
    zdim_ = (zmax_-zmin_)/zres_;
    voxels_=vector<vector<vector<Voxel>>>(xdim_, vector<vector<Voxel>>(ydim_, vector<Voxel>(zdim_,Voxel())));
    return true;
}

inline tuple<int,int,int> OccupancyGrid::getVoxelCoords(Vector3f point)
{
    int xv = floor((point(0)-xmin_)/xres_);
    int yv = floor((point(1)-ymin_)/yres_);
    int zv = floor((point(2)-zmin_)/zres_);
    return make_tuple(xv,yv,zv);
}

bool OccupancyGrid::validPoints(Vector3f point)
{
    float x = point(0);
    float y = point(1);
    float z = point(2);
    return !(x>=xmax_||y>=ymax_||z>=zmax_||x<=xmin_||y<=ymin_||z<=zmin_);
}
