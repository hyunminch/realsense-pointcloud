# realsense-pointcloud
A command-line app that can capture RGBD images from Intel RealSense D435i and perform registration on them based on RGB edge extraction, rotation estimation through D435i's IMU, Normal Distributions Transform (NDT), and Iterative Closest Point algorithm.

Our goal is to be able to create 3D models of indoor scenes with abundant RGB edges.

## Environment
Since librealsense2 only fully supports Linux platforms, so does this app.

## Dependencies
1. Install [librealsense2](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md) manually. (Install OpenGL2 / glfw3 while doing this.)
2. Install [PCL](http://pointclouds.org/downloads/) (version >= 1.9)

## Build
```bash
cmake .
make all
```

## Run
```bash
./rs-pcl [OPTIONS] NR_CLOUDS
```
### Options:
#### --all
capture and perform registration for NR_CLOUDS time using dynamic rotation estimation with the IMU of RealSense D435i.
#### --capture FILENAME
capture clouds for NR_CLOUDS time and save them to dataset/${FILENAME}-${CLOUD_IDX}.pcd CLOUD_IDX is given based on the order of capture
#### --registration FILENAME \[ROTATION_DEG\]
perform registration for NR_CLOUDS time on files named dataset/${FILENAME}-${CLOUD_IDX}.pcd using estimated rotation degree of ROTATION_DEG as initial guesses.
Default ROTATION_DEG: -30 degrees
#### --view FILENAME
view pointcloud saved at dataset/${FILENAME}.pcd
#### --help
print this help

### Examples:
1. Capture 3 point clouds and perform registration using dynamic rotation estimation
```bash
./rs-pcl --all 3
```
2. Capture 3 point clouds and save them to dataset/test-0.pcd, dataset/test-1.pcd, dataset/test-2.pcd
```bash
rs-pcl --capture test 3
```
3. Perform registration using default rotation estimation on 3 point clouds saved at dataset/test-0.pcd, dataset/test-1.pcd, dataset/test-2.pcd
```bash
rs-pcl --registration test 3
```
4. Perform registration using rotation degree of 45 on 3 point clouds saved at dataset/test-0.pcd, dataset/test-1.pcd, dataset/test-2.pcd
```bash
rs-pcl --registration test 45 3
```
5. View pointcloud saved at test.pcd
```bash
rs-pcl --view test
```
