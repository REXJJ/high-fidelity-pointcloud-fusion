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

struct Voxel
{
    Vector3f normal;
    Vector3f centroid;
    bool occupied;
    bool normal_found;
    int count;
    Voxel()
    {
        normal = {0,0,0};
        centroid = {0,0,0};
        count = 0;
        occupied = false;
        normal_found = false;
    }
};

class OccupancyGrid
{
    public:
    double xmin_,xmax_,ymin_,ymax_,zmin_,zmax_;
    double xres_,yres_,zres_;
    double xcenter_,ycenter_,zcenter_;
    int xdim_,ydim_,zdim_;
    int k_;
    vector<vector<vector<Voxel>>> voxels_;
    vector<vector<vector<Voxel>>> voxels_reorganized_;

    OccupancyGrid(){k_=0;xdim_=ydim_=zdim_=0;};
    void setDimensions(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax);
    void setResolution(float x,float y,float z);
    void setK(int k);
    bool construct();

    Voxel getVoxel(Vector3f point);
    tuple<int,int,int> getVoxelCoords(Vector3f point);
    tuple<int,int,int> getVoxelCoords(unsigned long long int id);

    unsigned long long int getHash(Vector3f point);
    unsigned long long int getHashId(int x,int y,int z);

    bool validCoords(int xid,int yid,int zid);
    bool validPoints(Vector3f point);

    vector<unsigned long long int> getNeighborHashes(unsigned long long int hash,int K=1);
    bool updateStates(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr normals);
    bool updateStates(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr normals, Eigen::Affine3d transform);
    bool downloadCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);
    bool downloadHQCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);
    bool downloadReorganizedCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,bool clean=false);
};

constexpr double ball_radius = 0.015;
constexpr double cylinder_radius = 0.001;
constexpr double downsample_radius = 0.005;

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

bool OccupancyGrid::updateStates(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr normals)
{
#pragma omp parallel for \ 
  default(none) \
  shared(normals) \
  num_threads(8)
    for(int p=0;p<normals->points.size();p++)
    {
        auto pt = normals->points[p];
        Vector3f point = {pt.x,pt.y,pt.z};
        Vector3f normal = {pt.normal[0],pt.normal[1],pt.normal[2]};
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(point);
        for(int i=-k_;i<=k_;i++)
            for(int j=-k_;j<=k_;j++)
            {
                for(int k=-k_;k<=k_;k++)
                {
                    if(validCoords(x+i,y+j,z+k)==false)
                        continue;
                    Voxel &voxel = voxels_[x+i][y+j][z+k];
                    voxel.normal+=normal;
                    voxel.normal=voxel.normal.normalized();
                    voxel.normal_found = true;
                }
            }
    }
    // std::cout<<"Points in cloud: "<<cloud->points.size()<<std::endl;
#pragma omp parallel for \ 
  default(none) \
  shared(cloud) \
  num_threads(8)
    for(int p=0;p<cloud->points.size();p++)
    {
        auto pt = cloud->points[p];
        Vector3f point = {pt.x,pt.y,pt.z};
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(point);
        for(int i=-k_;i<=k_;i++)
        for(int j=-k_;j<=k_;j++)
        for(int k=-k_;k<=k_;k++)
        {
            if(validCoords(x+i,y+j,z+k)==false)
                continue;
            Voxel& voxel = voxels_[x+i][y+j][z+k];
            Vector3f centroid = {xmin_+xres_*(x+i)+xres_/2.0,ymin_+yres_*(y+j)+yres_/2.0,zmin_+zres_*(z+k)+zres_/2.0};
            Vector3f normal = voxel.normal;
            if(voxel.normal_found)
            {
                Vector3f projected_points = projectPointToVector(point,centroid,normal);
                double distance_to_normal = (point - projected_points).norm();
                if(distance_to_normal<cylinder_radius)
                {
                   int count = voxel.count;
                   count++;
                   voxel.centroid = voxel.centroid + (projected_points-voxel.centroid)/count;
                   voxel.count = count;
                }
            }

        }
        if(validCoords(x,y,z))
        {
            Voxel& voxel = voxels_[x][y][z];
            voxel.occupied = true;      
        }
    }
    return true;
}

bool OccupancyGrid::updateStates(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr normals,Eigen::Affine3d transform)
{
#pragma omp parallel for \ 
  default(none) \
  shared(normals) \
  num_threads(8)
    for(int p=0;p<normals->points.size();p++)
    {
        auto pt = normals->points[p];
        Vector3f point = {pt.x,pt.y,pt.z};
        Vector3f normal = {pt.normal[0],pt.normal[1],pt.normal[2]};
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(point);
        for(int i=-k_;i<=k_;i++)
            for(int j=-k_;j<=k_;j++)
            {
                for(int k=-k_;k<=k_;k++)
                {
                    if(validCoords(x+i,y+j,z+k)==false)
                        continue;
                    Voxel &voxel = voxels_[x+i][y+j][z+k];
                    voxel.normal+=normal;
                    voxel.normal=voxel.normal.normalized();
                    voxel.normal_found = true;
                }
            }
    }
    // std::cout<<"Points in cloud: "<<cloud->points.size()<<std::endl;
#pragma omp parallel for \ 
  default(none) \
  shared(cloud,transform) \
  num_threads(8)
    for(int p=0;p<cloud->points.size();p++)
    {
        auto pt = cloud->points[p];
        auto trans_pt = pcl::transformPoint (pt,transform);
        Vector3f point = {trans_pt.x,trans_pt.y,trans_pt.z};
        int x,y,z;
        tie(x,y,z) = getVoxelCoords(point);
        for(int i=-k_;i<=k_;i++)
        for(int j=-k_;j<=k_;j++)
        for(int k=-k_;k<=k_;k++)
        {
            if(validCoords(x+i,y+j,z+k)==false)
                continue;
            Voxel& voxel = voxels_[x+i][y+j][z+k];
            Vector3f centroid = {xmin_+xres_*(x+i)+xres_/2.0,ymin_+yres_*(y+j)+yres_/2.0,zmin_+zres_*(z+k)+zres_/2.0};
            Vector3f normal = voxel.normal;
            if(voxel.normal_found)
            {
                Vector3f projected_points = projectPointToVector(point,centroid,normal);
                double distance_to_normal = (point - projected_points).norm();
                if(distance_to_normal<cylinder_radius)
                {
                   int count = voxel.count;
                   count++;
                   voxel.centroid = voxel.centroid + (projected_points-voxel.centroid)/count;
                   voxel.count = count;
                }
            }
        }
        if(validCoords(x,y,z))
        {
            Voxel& voxel = voxels_[x][y][z];
            voxel.occupied = true;      
        }
    }
    return true;
}

bool OccupancyGrid::downloadCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
    if(cloud==nullptr)
    {
        std::cout<<"Cloud not initialized.."<<std::endl;
        return false;
    }
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                {
                    Voxel voxel = voxels_[x][y][z];
                    if(voxel.occupied==true)
                    {
                        pcl::PointXYZRGBNormal pt;
                        pt.x = voxel.centroid(0);
                        pt.y = voxel.centroid(1);
                        pt.z = voxel.centroid(2);
                        pt.r = 0;
                        pt.g = 0;
                        pt.b = 0;
                        pt.normal[0] = voxel.normal(0);
                        pt.normal[1] = voxel.normal(1);
                        pt.normal[2] = voxel.normal(2);
                        cloud->points.push_back(pt);
                    }
                }
    std::cout<<"Points: "<<cloud->points.size()<<std::endl;
    cloud->height = 1;
    cloud->width = cloud->points.size();
    return true;
}

bool OccupancyGrid::downloadReorganizedCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,bool clean)
{
    voxels_reorganized_=vector<vector<vector<Voxel>>>(xdim_, vector<vector<Voxel>>(ydim_, vector<Voxel>(zdim_,Voxel())));
    if(cloud==nullptr)
    {
        std::cout<<"Cloud not initialized.."<<std::endl;
        return false;
    }
#pragma omp parallel for \ 
  default(none) \
  shared(cloud) \
  num_threads(8)
    for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                {
                    Voxel voxel = voxels_[x][y][z];
                    Voxel &voxel_new = voxels_reorganized_[x][y][z];
                    voxel_new.normal = voxel.normal;
                    voxel_new.centroid = voxel.centroid;
                    voxel_new.count  = voxel.count;
                    voxel_new.normal_found = voxel.normal_found;

                }
    std::cout<<"Stage One done.."<<std::endl;
 #pragma omp parallel for \ 
  default(none) \
  shared(clean,cloud) \
  num_threads(8)
   for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                {
                    Voxel &voxel = voxels_reorganized_[x][y][z];
                    if(voxels_[x][y][z].occupied==false)
                        continue;
                    if(clean==true&&voxel.count<100)
                        continue;
                    int xx,yy,zz;
                    tie(xx,yy,zz) = getVoxelCoords(voxel.centroid);
                    if(validCoords(xx,yy,zz))
                    {
                        Voxel &voxel_new = voxels_reorganized_[xx][yy][zz];
                        voxel_new.occupied = true;
                        voxel_new.normal+=voxel.normal;
                        voxel_new.normal = voxel_new.normal.normalized();
                        if(voxel_new.count==0)
                        {
                            voxel_new.centroid = voxel.centroid;
                        }
                        else
                        {
                            voxel_new.centroid+=voxel.centroid;
                            voxel_new.centroid/=2.0;
                            voxel_new.count++;
                        }

                    }
                }
    std::cout<<"Stage Two done.."<<std::endl;
  for(int x=0;x<xdim_;x++)
        for(int y=0;y<ydim_;y++)
            for(int z=0;z<zdim_;z++)
                {
                    Voxel voxel = voxels_reorganized_[x][y][z];
                    if(voxel.occupied==true)
                    {
                        pcl::PointXYZRGBNormal pt;
                        pt.x = voxel.centroid(0);
                        pt.y = voxel.centroid(1);
                        pt.z = voxel.centroid(2);
                        pt.r = 0;
                        pt.g = 0;
                        pt.b = 0;
                        pt.normal[0] = voxel.normal(0);
                        pt.normal[1] = voxel.normal(1);
                        pt.normal[2] = voxel.normal(2);
                        cloud->points.push_back(pt);
                    }
                }
    std::cout<<"Stage Three done.."<<std::endl;
    std::cout<<"Points: "<<cloud->points.size()<<std::endl;
    cloud->height = 1;
    cloud->width = cloud->points.size();
    return true;
}

bool OccupancyGrid::downloadHQCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
    if(cloud==nullptr)
    {
        std::cout<<"Cloud not initialized.."<<std::endl;
        return false;
    }
        for(int x=0;x<xdim_;x++)
            for(int y=0;y<ydim_;y++)
                for(int z=0;z<zdim_;z++)
                {
                    Voxel voxel = voxels_[x][y][z];
                    if(voxel.occupied==true&&voxel.count>100)
                    {
                        pcl::PointXYZRGBNormal pt;
                        pt.x = voxel.centroid(0);
                        pt.y = voxel.centroid(1);
                        pt.z = voxel.centroid(2);
                        pt.r = 0;
                        pt.g = 0;
                        pt.b = 0;
                        pt.normal[0] = voxel.normal(0);
                        pt.normal[1] = voxel.normal(1);
                        pt.normal[2] = voxel.normal(2);
                        cloud->points.push_back(pt);
                    }
                }
    std::cout<<"Points: "<<cloud->points.size()<<std::endl;
    cloud->height = 1;
    cloud->width = cloud->points.size();
    return true;
}

void OccupancyGrid::setK(int k)
{
    k_ = k;
}


void OccupancyGrid::setDimensions(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax)
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

inline unsigned long long int OccupancyGrid::getHash(Vector3f point)
{
    auto coords = getVoxelCoords(point);
    unsigned long long int hash = get<0>(coords);
    hash=hash<<40^(get<1>(coords)<<20)^(get<2>(coords));
    return hash;
}

inline unsigned long long int OccupancyGrid::getHashId(int x,int y,int z)
{
    unsigned long long int hash = x;
    hash = (hash<<40)^(y<<20)^z;
    return hash;
}

inline tuple<int,int,int> OccupancyGrid::getVoxelCoords(Vector3f point)
{
    int xv = floor((point(0)-xmin_)/xres_);
    int yv = floor((point(1)-ymin_)/yres_);
    int zv = floor((point(2)-zmin_)/zres_);
    return make_tuple(xv,yv,zv);
}

inline Voxel OccupancyGrid::getVoxel(Vector3f point)
{
    int x,y,z;
    tie(x,y,z) = getVoxelCoords(point);
    return voxels_[x][y][z];
}

inline tuple<int,int,int> OccupancyGrid::getVoxelCoords(unsigned long long int id)
{
    constexpr unsigned long long int mask = (1<<20)-1;
    unsigned long long int xid = id>>40;
    unsigned long long int yid = id>>20&mask;
    unsigned long long int zid = id&mask;
    return make_tuple(xid,yid,zid);
}

inline bool OccupancyGrid::validCoords(int xid,int yid,int zid)
{
    return xid<xdim_&&yid<ydim_&&zid<zdim_&&xid>=0&&yid>=0&&zid>=0;
}

bool OccupancyGrid::validPoints(Vector3f point)
{
    float x = point(0);
    float y = point(1);
    float z = point(2);
    return !(x>=xmax_||y>=ymax_||z>=zmax_||x<=xmin_||y<=ymin_||z<=zmin_);
}

vector<unsigned long long int> OccupancyGrid::getNeighborHashes(unsigned long long int hash,int K)
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
                    neighbors.push_back(getHashId(x+i,y+j,z+k));
            }
        }
    }
    return neighbors;
}
