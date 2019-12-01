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
#include "ndt.hpp"

#include "filters/edge_filter.hpp"

using pcl_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

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

int main(int argc, char * argv[]) try {
    int nr_frames = atoi(argv[1]);

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    auto clouds = get_clouds(pipe, nr_frames);

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense PCL PointCloud Example");
    // Construct an object to manage view state
    state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    auto edge_based_registration = new EdgeBasedRegistration();
    auto result = edge_based_registration->registration(clouds);

    while (app) {
        draw_pointcloud(app, app_state, {result});
    }

    // rgb_point_cloud_pointer pairwise_result(new rgb_point_cloud);
    // rgb_point_cloud_pointer sum(new rgb_point_cloud);

    // auto registration_scheme = new IncrementalICP();

    // auto incremental_registration_result = registration_scheme->registration(clouds);

    // while (app) {
    //     draw_pointcloud(app, app_state, {incremental_registration_result});
    // }

    return EXIT_SUCCESS;
} catch (const rs2::error & e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

    /*
    Eigen::Matrix4f global_transform = Eigen::Matrix4f::Identity(), pair_transform;
    
    for (int i = 1; i < nr_frames; i++) {
        std::cout << "result size: " << pairwise_result->size() << std:: endl;
        std::cout << "on " << i << std::endl;

        rgb_point_cloud_pointer temp(new point_cloud);

        pair_align(clouds[i - 1], clouds[i], temp, pair_transform, true);

        //transform current pair into the global transform
        pcl::transformPointCloud(*temp, *pairwise_result, global_transform);

        // update the global transform
        global_transform *= pair_transform;

        *sum += *pairwise_result;
    }
     */

    // std::cout << "converging succeeded " << cnt_success << " times." << std::endl;

    // pcl::io::savePCDFileBinary("capture.pcd", *incremental_registration_result);