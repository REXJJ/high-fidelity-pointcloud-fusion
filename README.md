# HighFidelityPointcloudFusion 

* Add the global frame name, the flange frame name, a bounding box and the directory in which you want to save the data in the launch file.

* roslaunch pointcloud_fusion pointcloud_fusion_node.launch

* Use the following services:
   - start : starts capturing the pointclouds and adds to the occupancy grid.
   - stop : stops capturing.
   - process : extracts pointcloud in the occupancy grid and stores it in the given directory. In addition, this also saves the metadata associated with the pointcloud.

* WARNING: Check the RAM usage the first time you run this node. 

* Parameters Values:
radius of cylinder: 1mm
radius of sphere: 15mm
