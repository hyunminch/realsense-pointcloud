#include <librealsense2/rs.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/organized_edge_detection.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include "utils.hpp"
#include "types.hpp"
#include "capture.hpp"
#include "visualizer.hpp"
#include "incremental_icp.hpp"
#include "edge_based_registration.hpp"
#include "pairwise_icp.hpp"
// #include "ndt.hpp"

#include "filters/edge_filter.hpp"

using pcl_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

float3 theta;

void pair_align(const rgb_point_cloud_pointer cloud_src, const rgb_point_cloud_pointer cloud_tgt, rgb_point_cloud_pointer output, Eigen::Matrix4f &final_transform, bool downsample = false) {
    rgb_point_cloud_pointer src(new rgb_point_cloud);
    rgb_point_cloud_pointer tgt(new rgb_point_cloud);
    pcl::VoxelGrid<rgb_point> grid;

    if (downsample) {
        grid.setLeafSize(0.2, 0.2, 0.2);

        grid.setInputCloud(cloud_src);
        grid.filter(*src);

        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    } else {
        src = cloud_src;
        tgt = cloud_tgt;
    }

    rgb_normal_point_cloud::Ptr points_with_normals_src(new rgb_normal_point_cloud);
    rgb_normal_point_cloud::Ptr points_with_normals_tgt(new rgb_normal_point_cloud);

    pcl::NormalEstimation<rgb_point, rgb_normal_point> norm_est;
    pcl::search::KdTree<rgb_point>::Ptr tree(new pcl::search::KdTree<rgb_point>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(30);

    norm_est.setInputCloud(src);
    norm_est.compute(*points_with_normals_src);
    pcl::copyPointCloud(*src, *points_with_normals_src);

    norm_est.setInputCloud(src);
    norm_est.compute(*points_with_normals_tgt);
    pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

    pcl::IterativeClosestPointNonLinear<rgb_normal_point, rgb_normal_point> icp;
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1);
    icp.setInputSource(points_with_normals_src);
    icp.setInputTarget(points_with_normals_tgt);

    icp.setMaximumIterations(20);
    icp.align(*points_with_normals_src);
    Eigen::Matrix4f Ti = icp.getFinalTransformation();

    // Get the transformation from target to source
    Eigen::Matrix4f target_to_source = Ti.inverse();
    
    // Transform target back in source frame
    pcl::transformPointCloud(*cloud_tgt, *output, target_to_source);

    // add the source to the transformed target
    *output += *cloud_src;

    final_transform = target_to_source;
}

void capture(const std::string prefix, int frames) {
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 6);
    cfg.enable_stream(RS2_STREAM_COLOR,1280, 720, RS2_FORMAT_BGR8, 6);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 6);

    pipe.start(cfg);

    // pipe.start(cfg, [&](rs2::frame frame) {
    //     // Cast the frame that arrived to motion frame
    //     auto motion = frame.as<rs2::motion_frame>();
    //     // If casting succeeded and the arrived frame is from gyro stream
    //     if (motion && motion.get_profile() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
    //         // Get the timestamp that arrived to motion frame
    //         double ts = motion.get_timestamp();
    //         // Get gyro measures
    //         rs2_vector gyro_data = motion.get_motion_data();
    //         // Call function that computes the angle of motion based on the retrieved measures
    //         algo.process_gyro(gyro_data, ts);
    //     }

    //     if (motion && motion.get_profile() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
    //         // Get accelerometer measures
    //         rs2_vector accel_data = motion.get_motion_data();
    //         // Call function that computes the angle of motion based on the retrieved measures
    //         algo.process_accel(accel_data);
    //     }
    // });

    auto pair = get_clouds(pipe, frames);
    auto clouds = pair.first;
    auto thetas = pair.second;
    for (int frame = 0; frame < frames; frame++) {
        pcl::io::savePCDFileBinary("dataset/" + prefix + "-" + std::to_string(frame), *clouds[frame]);
    }

    pipe.stop();
}

void registration(const std::string prefix, int frames) {
    std::vector<rgb_point_cloud_pointer> clouds;

    for (int frame = 0; frame < frames; frame++) {
        rgb_point_cloud_pointer cloud_ptr(new rgb_point_cloud);
        pcl::io::loadPCDFile("dataset/" + prefix + "-" + std::to_string(frame), *cloud_ptr);
        clouds.push_back(cloud_ptr);
    }

//    auto incremental_icp = new IncrementalICP();
//    auto result = incremental_icp->registration(clouds);

    auto edge_based_registration = new EdgeBasedRegistration();
    auto result = edge_based_registration->registration(clouds);

    pcl::io::savePCDFileBinary("dataset/" + prefix + "-registration", *result);

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense PCL PointCloud Example");
    // Construct an object to manage view state
    state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    while (app) {
        draw_pointcloud(app, app_state, {result});
    }
}

void viewer(std::string name) {
    rgb_point_cloud_pointer cloud(new rgb_point_cloud);
    pcl::io::loadPCDFile("dataset/" + name, *cloud);

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense PCL PointCloud Example");
    // Construct an object to manage view state
    state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    while (app) {
        draw_pointcloud(app, app_state, {cloud});
    }
}

void capture_and_registration(int frames) {
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    auto pair = get_clouds(pipe, frames);
    auto clouds = pair.first;
    auto thetas = pair.second;

    auto incremental_icp = new IncrementalICP();
    // auto result = incremental_icp->registration(clouds);

    auto edge_based_registration = new EdgeBasedRegistration(thetas);
    // edge_based_registration->set_thetas(thetas);
    auto result = edge_based_registration->registration(clouds);

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense PCL PointCloud Example");
    // Construct an object to manage view state
    state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    while (app) {
        draw_pointcloud(app, app_state, {result});
    }
}

int main(int argc, char *argv[]) try {
    if (argc == 1) {
        capture_and_registration(3);

        return 0;
    } else if (strcmp(argv[1], "--capture") == 0) {
        std::string dataset_prefix = argv[2];
        int frames = atoi(argv[3]);

        capture(dataset_prefix, frames);

        return 0;
    } else if (strcmp(argv[1], "--registration") == 0) {
        std::string dataset_prefix = argv[2];
        int frames = atoi(argv[3]);

        registration(dataset_prefix, frames);

        return 0;
    } else if (strcmp(argv[1], "--view") == 0)  {
        std::string name = argv[2];
        viewer(name);

        return 0;
    } else {
        int frames = atoi(argv[1]);

        capture_and_registration(frames);

        return 0;
    }
} catch (const rs2::error & e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
