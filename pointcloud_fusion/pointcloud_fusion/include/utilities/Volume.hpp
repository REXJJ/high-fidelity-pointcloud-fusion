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

using namespace std;
using namespace pcl;

struct Voxel
{
    vector<pcl::PointXYZRGB> pts;
    vector<pcl::Normal> normals;
    int view;
    bool good;
    Voxel(pcl::PointXYZRGB pt)
    {
        pts.push_back(pt);
        view=0;
        good = false;
    }
    Voxel(pcl::PointXYZRGB pt,pcl::Normal normal)
    {
        pts.push_back(pt);
        view=0;
        good = false;
        normals.push_back(normal);
    }
};

class VoxelVolume
{
    unordered_map<unsigned long long int,vector<vector<float>>> lut_;
    public:
    vector<unsigned long long int> occupied_cells_;
    double xmin_,xmax_,ymin_,ymax_,zmin_,zmax_;
    double xcenter_,ycenter_,zcenter_;
    double xdelta_,ydelta_,zdelta_;
    double voxel_size_;
    int xdim_,ydim_,zdim_;
    unsigned long long int hsize_;
    vector<vector<vector<Voxel*>>> voxels_;
    VoxelVolume(){};
    void setDimensions(double xmin,double xmax,double ymin,double ymax, double zmin, double zmax);
    void setResolution(double xdelta, double ydelta, double zdelta);
    void setVolumeSize(int xdim,int ydim,int zdim);
    bool constructVolume();
    template<typename PointT> bool addPointCloud(typename pcl::PointCloud<PointT>::Ptr pointcloud);
    unsigned long long int getHash(float x,float y,float z);
    unsigned long long int getHashId(int x,int y,int z);
    tuple<int,int,int> getVoxel(float x,float y,float z);
    ~VoxelVolume();
    bool integratePointCloud(pcl::PointCloud<PointXYZRGB>::Ptr cloud);
    bool integratePointCloud(pcl::PointCloud<PointXYZRGB>::Ptr cloud,pcl::PointCloud<pcl::Normal>::Ptr normal);
    bool validPoints(float x,float y,float z);
    tuple<int,int,int> getVoxelCoords(unsigned long long int id);
    bool validCoords(int xid,int yid,int zid);
    vector<unsigned long long int> getNeighborHashes(unsigned long long int hash,int K=1);
};

VoxelVolume::~VoxelVolume()
{
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                if(voxels_[x][y][z])
                    delete voxels_[x][y][z];
}

void VoxelVolume::setDimensions(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax)
{
    xmin_=xmin;
    xmax_=xmax;
    ymin_=ymin;
    ymax_=ymax;
    zmin_=zmin;
    zmax_=zmax;
    xcenter_=xmin_+(xmax_-xmin_)/2.0;
    ycenter_=ymin_+(ymax_-ymin_)/2.0;
    zcenter_=zmin_+(zmax_-zmin_)/2.0;
}

void VoxelVolume::setResolution(double xdelta,double ydelta,double zdelta)
{
    xdelta_=xdelta;
    ydelta_=ydelta;
    zdelta_=zdelta;
    voxel_size_=xdelta_*ydelta_*zdelta_;
    xdim_=(xmax_-xmin_)/xdelta_;
    ydim_=(ymax_-ymin_)/ydelta_;
    zdim_=(zmax_-zmin_)/zdelta_;
    hsize_=xdim_*ydim_*zdim_;
}

void VoxelVolume::setVolumeSize(int xdim,int ydim,int zdim)
{
    xdim_=xdim;
    ydim_=ydim;
    zdim_=zdim;
    xdelta_=(xmax_-xmin_)/xdim;
    ydelta_=(ymax_-ymin_)/ydim;
    zdelta_=(zmax_-zmin_)/zdim;
    voxel_size_=xdelta_*ydelta_*zdelta_;
    hsize_=xdim_*ydim_*zdim_;
}

bool VoxelVolume::constructVolume()
{
    voxels_=vector<vector<vector<Voxel*>>>(xdim_, vector<vector<Voxel*>>(ydim_, vector<Voxel*>(zdim_,nullptr)));
    return true;
}

template<typename PointT> bool VoxelVolume::addPointCloud(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    return true;
}

inline unsigned long long int VoxelVolume::getHash(float x,float y,float z)
{
    auto coords = getVoxel(x,y,z);
    unsigned long long int hash = get<0>(coords);
    hash=hash<<40^(get<1>(coords)<<20)^(get<2>(coords));
    return hash;
}

inline unsigned long long int VoxelVolume::getHashId(int x,int y,int z)
{
    unsigned long long int hash = x;
    hash = (hash<<40)^(y<<20)^z;
    return hash;
}

inline tuple<int,int,int> VoxelVolume::getVoxel(float x,float y,float z)
{
    int xv = floor((x-xmin_)/xdelta_);
    int yv = floor((y-ymin_)/ydelta_);
    int zv = floor((z-zmin_)/zdelta_);
    return make_tuple(xv,yv,zv);
}

inline tuple<int,int,int> VoxelVolume::getVoxelCoords(unsigned long long int id)
{
    constexpr unsigned long long int mask = (1<<20)-1;
    unsigned long long int xid = id>>40;
    unsigned long long int yid = id>>20&mask;
    unsigned long long int zid = id&mask;
    return make_tuple(xid,yid,zid);
}

inline bool VoxelVolume::validCoords(int xid,int yid,int zid)
{
    return xid<xdim_&&yid<ydim_&&zid<zdim_&&xid>=0&&yid>=0&&zid>=0;
}

bool VoxelVolume::integratePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    for(int i=0;i<cloud->points.size();i++)
    {
        pcl::PointXYZRGB pt = cloud->points[i];
        if(validPoints(pt.x,pt.y,pt.z)==false)
            continue;
        auto coords = getVoxel(pt.x,pt.y,pt.z);
        int x = get<0>(coords);
        int y = get<1>(coords);
        int z = get<2>(coords);
        auto hash = getHashId(x,y,z);
        if(voxels_[x][y][z]==nullptr)
        {
            occupied_cells_.push_back(hash);
            Voxel *voxel = new Voxel(pt);
            voxels_[get<0>(coords)][get<1>(coords)][get<2>(coords)] = voxel;
        }
        else
        {
            Voxel *voxel = voxels_[x][y][z];
            voxel->pts.push_back(pt);
        }
    }

}

bool VoxelVolume::integratePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    for(int i=0;i<cloud->points.size();i++)
    {
        pcl::PointXYZRGB pt = cloud->points[i];
        pcl::Normal normal = normals->points[i];
        if(validPoints(pt.x,pt.y,pt.z)==false)
            continue;
        auto coords = getVoxel(pt.x,pt.y,pt.z);
        int x = get<0>(coords);
        int y = get<1>(coords);
        int z = get<2>(coords);
        auto hash = getHashId(x,y,z);
        if(voxels_[x][y][z]==nullptr)
        {
            occupied_cells_.push_back(hash);
            Voxel *voxel = new Voxel(pt,normal);
            voxels_[get<0>(coords)][get<1>(coords)][get<2>(coords)] = voxel;
        }
        else
        {
            Voxel *voxel = voxels_[x][y][z];
            voxel->pts.push_back(pt);
            voxel->normals.push_back(normal);
        }
    }

}

bool VoxelVolume::validPoints(float x,float y,float z)
{
    return !(x>=xmax_||y>=ymax_||z>=zmax_||x<=xmin_||y<=ymin_||z<=zmin_);
}

vector<unsigned long long int> VoxelVolume::getNeighborHashes(unsigned long long int hash,int K)
{
    double x,y,z;
    tie(x,y,z) = getVoxelCoords(hash);
    vector<unsigned long long int> neighbors;
    for(int i=-K;i<=K;i++)
    {
        for(int j=-K;j<=K;j++)
        {
            for(int k=-K;k<=K;k++)
            {
                if(i==j==k==0)
                    continue;
                if(validCoords(x+i,y+j,z+k))
                    if(voxels_[x+i][y+j][z+k]!=nullptr)
                        neighbors.push_back(getHashId(x+i,y+j,z+k));
            }
        }
    }
    return neighbors;
}
