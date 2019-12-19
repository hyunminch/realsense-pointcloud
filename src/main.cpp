#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

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
#include "capture_opencv.hpp"
#include "visualizer.hpp"
#include "incremental_icp.hpp"
#include "edge_extractor.hpp"
#include "ndt_edge_based_registration.hpp"
#include "icp_edge_based_registration.hpp"

void capture(const std::string prefix, int frames) {
    rs2::pipeline pipe;
    rs2::config cfg;

    pipe.start(cfg);

    auto point_cloud = get_clouds_new(pipe, frames, prefix);
    pipe.stop();
    pcl::io::savePCDFile(fn_result(prefix), *point_cloud);

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense PCL PointCloud Example");
    // Construct an object to manage view state
    state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    while (app) {
        draw_pointcloud(app, app_state, {point_cloud});
    }

//    std::vector<rgb_point_cloud_pointer> clouds;
//    std::vector<Eigen::Matrix4f> transformations;
//    for (auto pair : pairs) {
//        clouds.push_back(pair.first);
//        transformations.push_back(pair.second);
//    }
//    auto clouds = pair.first;
//    auto thetas = pair.second;
//    for (int frame = 0; frame < frames; frame++)
//        pcl::io::savePCDFileBinary("dataset/" + prefix + "-" + std::to_string(frame) + ".pcd", *clouds[frame]);

}

void edges(const std::string filename) {
    rgb_point_cloud_pointer cloud_ptr(new rgb_point_cloud);
    pcl::io::loadPCDFile("dataset/" + filename, *cloud_ptr);
    
    auto result = extract_edge_features(cloud_ptr);

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

void registration(const std::string prefix, RegistrationScheme *scheme, int frames) {
    std::vector<rgb_point_cloud_pointer> clouds;

    for (int frame = 0; frame < frames; frame++) {
        rgb_point_cloud_pointer cloud_ptr(new rgb_point_cloud);
        pcl::io::loadPCDFile("dataset/" + prefix + "-" + std::to_string(frame) + ".pcd", *cloud_ptr);
        clouds.push_back(cloud_ptr);
    }

    auto result = scheme->registration(clouds);

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
    pcl::io::loadPCDFile("dataset/" + name + ".pcd", *cloud);

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

void capture_and_registration(int frames, std::string icp_based_filename) {
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Start streaming with default recommended configuration
    pipe.start();
    auto pair = get_clouds(pipe, frames);
    pipe.stop();

    auto clouds = pair.first;
    auto thetas = pair.second;

    auto icp_edge_based_registration = new ICPEdgeBasedRegistration(thetas);

    auto icp_result = icp_edge_based_registration->registration(clouds);

    pcl::io::savePCDFileBinary("dataset/" + icp_based_filename + ".pcd", *icp_result);
}

void help() {
    std::cout << "Usage: rs-pcl [OPTION] NR_CLOUDS..." << std::endl
              << "Capture, perform registration, or do both for NR_CLOUDS time." << std::endl
              << "Example: rs-pcl --all 4" << std::endl
              << std::endl
              << "Options:" << std::endl
              << "  --all" << std::endl
              << "      capture and perform registration for NR_CLOUDS time" << std::endl
              << "      using dynamic rotation estimation with the IMU of RealSense D435i." << std::endl
              << "  --capture FILENAME" << std::endl
              << "      capture clouds for NR_CLOUDS time and save them to" << std::endl
              << "      dataset/${FILENAME}-${CLOUD_IDX}.pcd" << std::endl
              << "      CLOUD_IDX is given based on the order of capture" << std::endl
              << "  --registration FILENAME [ROTATION_DEG]" << std::endl  
              << "      perform registration for NR_CLOUDS time on files named" << std::endl
              << "      dataset/${FILENAME}-${CLOUD_IDX}.pcd" << std::endl
              << "      using estimated rotation degree of ROTATION_DEG as initial guesses." << std::endl
              << "      Default ROTATION_DEG: -30 degrees" << std::endl
              << "  --view FILENAME" << std::endl
              << "      view pointcloud saved at dataset/${FILENAME}.pcd" << std::endl
              << "  --help" << std:: endl
              << "      print this help"
              << std::endl
              << std::endl
              << "Examples:" << std::endl
              << "  capture 3 point clouds and perform registration using dynamic rotation estimation" << std::endl
              << "  $ rs-pcl --all 3" << std::endl
              << std::endl
              << "  capture 3 point clouds and save them to" << std::endl 
              << "    dataset/test-0.pcd, dataset/test-1.pcd, dataset/test-2.pcd" << std::endl
              << "  $ rs-pcl --capture test 3" << std::endl
              << std::endl
              << "  extract edges from a given pointcloud saved at" << std::endl
              << "    dataset/testcase.pcd"
              << "  $ rs-pcl --edges testcase.pcd" << std::endl
              << std::endl
              << "  perform registration using default rotation estimation on 3 point clouds saved at" << std::endl
              << "    dataset/test-0.pcd, dataset/test-1.pcd, dataset/test-2.pcd" << std::endl
              << "  $ rs-pcl --registration test 3" << std::endl
              << std::endl
              << "  perform registration using rotation degree of 45 on 3 point clouds saved at" << std::endl
              << "    dataset/test-0.pcd, dataset/test-1.pcd, dataset/test-2.pcd" << std::endl
              << "  $ rs-pcl --registration test 45 3" << std::endl
              << std::endl
              << "  view pointcloud saved at test.pcd" << std::endl
              << "  $ rs-pcl --view test" << std::endl
              << std::endl;
}

int main(int argc, char *argv[]) try {
    cv::namedWindow("Display Image");

    if (argc == 1) {
        help();

        return EXIT_FAILURE;
    } else if (strcmp(argv[1], "--capture") == 0 && argc == 4) {
        std::string dataset_prefix = argv[2];
        int frames = atoi(argv[3]);

        capture(dataset_prefix, frames);

        return 0;
    } else if (strcmp(argv[1], "--edges") == 0 && argc == 3) {
        std::string filename = argv[2];
        edges(filename);
        
        return 0;
    } else if (strcmp(argv[1], "--registration") == 0 && argc == 4) {
        std::string dataset_prefix = argv[2];
        int frames = atoi(argv[3]);

        auto edge_based_registration = new NDTEdgeBasedRegistration();
        registration(dataset_prefix, edge_based_registration, frames);

        return 0;
    } else if (strcmp(argv[1], "--registration") == 0 && argc == 5) {
        std::string dataset_prefix = argv[2];
        int rotation_deg = atoi(argv[3]);
        float rads = (rotation_deg / 180.0) * M_PI;
        int frames = atoi(argv[4]);

        auto edge_based_registration = new NDTEdgeBasedRegistration(rads);
        registration(dataset_prefix, edge_based_registration, frames);

        return 0;
    } else if (strcmp(argv[1], "--view") == 0 && argc == 3)  {
        std::string name = argv[2];
        viewer(name);

        return 0;
    } else if (strcmp(argv[1], "--all") == 0 && argc == 4) {
        int frames = atoi(argv[2]);
        std::string icp_based_filename = argv[3];

        capture_and_registration(frames, icp_based_filename);
        
        return 0;
    } else {
        help();
        return EXIT_FAILURE;
    }
} catch (const rs2::error & e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
