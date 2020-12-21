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

inline Vector3f projectPointToVector(Vector3f pt, Vector3f norm_pt, Vector3f n)
{
    Vector3f d_xyz = n*kBballRadius;
    Vector3f a = norm_pt - d_xyz;
    Vector3f b = norm_pt + d_xyz;
    Vector3f ap = (a-pt);
    Vector3f ab = (a-b);
    Vector3f p = a - (ap.dot(ab)/ab.dot(ab))*ab;
    return p;
}

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
    int count;
    VoxelInfo()
    {
        normal_found = false;
        count = 0;
        centroid = {0,0,0};
    }
};

class OccupancyGrid
{
    private:
        vector<int> dx,dy,dz;
    public:
        double xmin_,xmax_,ymin_,ymax_,zmin_,zmax_;
        double xres_,yres_,zres_;
        int xdim_,ydim_,zdim_;
        int k_;
        vector<vector<vector<Voxel>>> voxels_;
        int counter;
        bool state_changed;
        OccupancyGrid(){k_=2;xdim_=ydim_=zdim_=0;counter=0;state_changed=false;};//TODO: Get k_ at compile time.
        void setDimensions(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax);
        void setResolution(float x,float y,float z);
        bool construct();
        tuple<int,int,int> getVoxelCoords(Vector3f point);
        tuple<int,int,int> getVoxelCoords(unsigned long long int hash);
        unsigned long long int getHashId(int x,int y,int z);
        bool validPoints(Vector3f point);
        bool validCoord(int x,int y,int z);
        bool addPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        template<int N> bool updateStates();
        bool clearVoxels();
        bool download(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        bool download(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);
        bool downloadHQ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        bool setK(int k);
        unordered_set<unsigned long long int> unprocessed_data_;
        Vector3f getVoxelCenter(int x,int y,int z)
        {
            Vector3f centroid = {xmin_+xres_*(x)+xres_/2.0,ymin_+yres_*(y)+yres_/2.0,zmin_+zres_*(z)+zres_/2.0};
            return centroid;
        }
};

bool OccupancyGrid::setK(int k)
{
    k_ = k;
    for(int i=-k_;i<=k_;i++)
        for(int j=-k_;j<=k_;j++)
            for(int k=-k_;k<=k_;k++)
            {
                dx.push_back(i);
                dy.push_back(j);
                dz.push_back(k);
            }
    std::cout<<"D vectors: "<<std::endl;
    std::cout<<"Size of D vectors: "<<dx.size()<<std::endl;
    for(auto x:dx)
        std::cout<<x<<" "<<std::endl;
    std::cout<<"-------------------------"<<std::endl;
    for(auto y:dy)
        std::cout<<y<<" "<<std::endl;
    std::cout<<"-------------------------"<<std::endl;
    for(auto z:dz)
        std::cout<<z<<" "<<std::endl;
    std::cout<<"-------------------------"<<std::endl;
}

inline unsigned long long int OccupancyGrid::getHashId(int x,int y,int z)
{
    unsigned long long int hash = x;
    hash = (hash<<40)^(y<<20)^z;
    return hash;
}

inline tuple<int,int,int> OccupancyGrid::getVoxelCoords(unsigned long long int id)
{
    constexpr unsigned long long int mask = (1<<20)-1;
    unsigned long long int xid = id>>40;
    unsigned long long int yid = id>>20&mask;
    unsigned long long int zid = id&mask;
    return make_tuple(xid,yid,zid);
}

bool OccupancyGrid::clearVoxels()
{
    state_changed = true;
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                {
                    if(voxels_[x][y][z].occupied==true)
                    {
                        VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxels_[x][y][z].data);
                        delete data;
                        voxels_[x][y][z].data = nullptr;
                        voxels_[x][y][z].occupied = false;
                    }
                }
                std::cout<<"All voxels cleared..."<<std::endl;
}

bool OccupancyGrid::addPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    if(cloud==nullptr)
        return false;
    state_changed = true;
    // #pragma omp parallel for \ 
    //     default(none) \
    //         shared(cloud) \
    //         num_threads(4)
    for(int p=0;p<cloud->points.size();p++)
    {
        auto pt = cloud->points[p];
        Vector3f point = {pt.x,pt.y,pt.z};
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(point);
        if(validPoints(point)==false)
            continue;
        unsigned long long int hash = getHashId(x,y,z);
        Voxel& voxel = voxels_[x][y][z];
        Vector3f ptv = {pt.x,pt.y,pt.z};
        if(voxel.occupied==true)
        {
            VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxel.data);
            if(data->normal_found==false)
                data->buffer.push_back(ptv);
            else
                if(unprocessed_data_.find(hash)!=unprocessed_data_.end())
                    unprocessed_data_.erase(hash);
        }
        else
        {
            voxel.occupied = true;
            unprocessed_data_.insert(hash);
            VoxelInfo* data = new VoxelInfo();
            // data->buffer.reserve(1000);
            data->buffer.push_back(ptv);
            unprocessed_data_.insert(hash);
            voxels_[x][y][z].data = reinterpret_cast<void*>(data);
            //TODO: Allocate 1000 before hand.
        }
        //TODO: Check state before doing this..
        //TODO: This is a big bottle neck. 
        for(int d=0;d<dx.size();d++)
        {
            int i = dx[d];
            int j = dy[d];
            int k = dz[d];
            if(validCoord(x+i,y+j,z+k))
            {
                // std::cout<<x+i<<" "<<y+j<<" "<<z+k<<std::endl;
                Voxel voxel_neighbor = voxels_[x+i][y+j][z+k];
                if(voxel_neighbor.occupied==true)
                {
                    // std::cout<<"Here"<<std::endl;
                    VoxelInfo* neighbor_data = reinterpret_cast<VoxelInfo*>(voxel_neighbor.data);
                    if(neighbor_data->normal_found==true)
                    {
                        // std::cout<<"Here 2"<<std::endl;
                        Vector3f centroid = {xmin_+xres_*(x+i)+xres_/2.0,ymin_+yres_*(y+j)+yres_/2.0,zmin_+zres_*(z+k)+zres_/2.0};
                        Vector3f projected_points = projectPointToVector(ptv,centroid,neighbor_data->normal);
                        double distance_to_normal = (ptv - projected_points).norm();
                        // std::cout<<distance_to_normal<<std::endl;
                        if(distance_to_normal<kCylinderRadius)
                        {
                            neighbor_data->count++;
                            neighbor_data->centroid = neighbor_data->centroid + (projected_points-neighbor_data->centroid)/neighbor_data->count;
                        }
                    }
                }
            }
        }

    }
    return true;
}

template<int N> bool OccupancyGrid::updateStates()
{
    state_changed = false;
    std::vector<unsigned long long int> keys;
    for(auto key:unprocessed_data_)
        keys.push_back(key);
    std::cout<<"Total Voxels: "<<keys.size()<<std::endl;
    //TODO: Use hashing.
#pragma omp parallel for \ 
    default(none) \
    shared(keys) \
        num_threads(N)
    for (int key=0;key<keys.size();key++)
    {
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(keys[key]);
        Voxel& voxel = voxels_[x][y][z];
        if(voxel.occupied==true)
        {
            VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxel.data);
            int total = 0;
            //TODO: Replace this hideous loop with something better: templates with par for might be a good option.
            bool neighbors_done = true;
            for(int i=-k_;i<=k_;i++)
                for(int j=-k_;j<=k_;j++)
                    for(int k=-k_;k<=k_;k++)
                    {
                        if(validCoord(x+i,y+j,z+k))
                        {
                            Voxel voxel_neighbor = voxels_[x+i][y+j][z+k];
                            if(voxel_neighbor.occupied==true)
                            {
                                VoxelInfo* neighbor_data = reinterpret_cast<VoxelInfo*>(voxel_neighbor.data);
                                total+=neighbor_data->buffer.size();
                                if(neighbor_data->normal_found==false)
                                    neighbors_done = false;
                            }
                        }
                    }

            // std::cout<<"Total: "<<total<<std::endl;

            if(total>1000&&data->normal_found==false)
            {
                // std::cout<<"Reaching Here.."<<std::endl;
                // std::cout<<"Total: "<<total<<std::endl;
                Eigen::MatrixXd lhs (total, 3);
                Eigen::VectorXd rhs (total);
                //TODO: Replace this hideos loop with something better: templates with par for might be a good option.
                int counter = 0;
                for(int i=-k_;i<=k_;i++)
                    for(int j=-k_;j<=k_;j++)
                        for(int k=-k_;k<=k_;k++)
                        {
                            if(validCoord(x+i,y+j,z+k))
                            {
                                // std::cout<<x+i<<" "<<y+j<<" "<<z+k<<std::endl;
                                Voxel voxel_neighbor = voxels_[x+i][y+j][z+k];
                                if(voxel_neighbor.occupied==false)
                                    continue;
                                VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxel_neighbor.data);
                                for(auto x:data->buffer)
                                {
                                    lhs(counter,0) = x(0);
                                    lhs(counter,1) = x(1);
                                    lhs(counter,2) = 1.0;
                                    rhs(counter++) = -1.0*x(2);
                                }
                            }
                        }

                // std::cout<<"Total: "<<total<<" "<<counter<<std::endl;

                Eigen::Vector3d params = lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);//Jacobi SVD
                Eigen::Vector3f normal (params(0), params(1), 1.0);
                auto length = normal.norm();
                normal /= length;
                data->normal = normal;
                data->normal_found = true;
                Vector3f centroid = {xmin_+xres_*(x)+xres_/2.0,ymin_+yres_*(y)+yres_/2.0,zmin_+zres_*(z)+zres_/2.0};
                for(auto pt:data->buffer)
                {
                    Vector3f projected_points = projectPointToVector(pt,centroid,normal);
                    double distance_to_normal = (pt - projected_points).norm();
                    if(distance_to_normal<kCylinderRadius)
                    {
                        data->count++;
                        data->centroid = data->centroid + (projected_points-data->centroid)/data->count;
                    }
                }
                if(neighbors_done)
                {
                    data->buffer.clear();
                    data->buffer.shrink_to_fit();
                    VoxelInfo* new_data = new VoxelInfo();
                    new_data->normal = data->normal;
                    new_data->centroid = data->centroid;
                    new_data->count = data->count;
                    new_data->normal_found = data->normal_found;
                    delete data;
                    voxels_[x][y][z].data = reinterpret_cast<void*>(new_data);
                }
            }
        }
    }
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
                    if(data->normal_found==false)
                        continue;
                    pcl::PointXYZRGB pt;
                    auto point = data->centroid;
                    // auto point = getVoxelCenter(x,y,z);
                    pt.x = point(0);
                    pt.y = point(1);
                    pt.z = point(2);
                    cloud->points.push_back(pt);
                }
    std::cout<<"Points: "<<cloud->points.size()<<std::endl;
}

bool OccupancyGrid::downloadHQ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    if(cloud==nullptr)
        return false;
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                if(voxels_[x][y][z].occupied)
                {
                    VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxels_[x][y][z].data);
                    if(data->normal_found==false)
                        continue;
                    if(data->count<kGoodPointsThreshold)
                        continue;
                    pcl::PointXYZRGB pt;
                    auto point = data->centroid;
                    // auto point = getVoxelCenter(x,y,z);
                    pt.x = point(0);
                    pt.y = point(1);
                    pt.z = point(2);
                    cloud->points.push_back(pt);
                }
    std::cout<<"Points: "<<cloud->points.size()<<std::endl;
}

bool OccupancyGrid::download(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
    if(cloud==nullptr)
        return false;
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                if(voxels_[x][y][z].occupied)
                {
                    VoxelInfo* data = reinterpret_cast<VoxelInfo*>(voxels_[x][y][z].data);
                    if(data->normal_found==false)
                        continue;
                    pcl::PointXYZRGBNormal pt;
                    // auto point = data->centroid;
                    auto point = getVoxelCenter(x,y,z);
                    pt.x = point(0);
                    pt.y = point(1);
                    pt.z = point(2);
                    pt.normal[0] = data->normal(0);
                    pt.normal[1] = data->normal(1);
                    pt.normal[2] = data->normal(2);
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
    voxels_=vector<vector<vector<Voxel>>>(xdim_+1, vector<vector<Voxel>>(ydim_+1, vector<Voxel>(zdim_+1,Voxel())));
    return true;
}

inline tuple<int,int,int> OccupancyGrid::getVoxelCoords(Vector3f point)
{
    //TODO: Verify if this is fine. 
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

inline bool OccupancyGrid::validCoord(int x,int y,int z)
{
    return (x>=0&&y>=0&&z>=0&&x<xdim_&&y<ydim_&&z<zdim_);
}
