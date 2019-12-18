#include <librealsense2/rs.hpp>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>

#include "types.hpp"
#include "utils.hpp"
#include "rotation_estimator.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

std::pair<std::vector<cv::KeyPoint>, Mat> get_keypoints(int frame, Mat input) {
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
    Mat input_converted(input.size(), CV_8UC1);
    cv::cvtColor(input, input_converted, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    Mat descriptors;

    detector->detectAndCompute(input, noArray(), keypoints, descriptors);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);

    const char* corners_window = ("Corner " + std::to_string(frame)).c_str();
    namedWindow(corners_window);
    imshow(corners_window, output);

    return std::make_pair(keypoints, descriptors);
}

std::pair<std::vector<Point2f>, std::vector<Point2f>> get_keypoints_twoframes(rs2::video_frame color_frame1, rs2::video_frame color_frame2) {
    Mat input_1(Size(1280, 720), CV_8UC3, (void*)color_frame1.get_data(), Mat::AUTO_STEP);
    Mat input_2(Size(1280, 720), CV_8UC3, (void*)color_frame2.get_data(), Mat::AUTO_STEP);

    auto result1 = get_keypoints(0, input_1);
    auto result2 = get_keypoints(1, input_2);

    auto keypoints_1 = result1.first;
    auto keypoints_2 = result2.first;
    auto descriptors1 = result1.second;
    auto descriptors2 = result2.second;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.3f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    std::sort(good_matches.begin(), good_matches.end());

    // Draw top matches
    Mat im_matches;
    drawMatches(input_1, keypoints_1, input_2, keypoints_2, good_matches, im_matches);
    imwrite("matches.jpg", im_matches);

    // Extract location of good matches
    std::vector<Point2f> points1, points2;

    for (size_t i = 0; i < good_matches.size(); i++) {
        auto point_1 = keypoints_1[good_matches[i].queryIdx].pt;
        auto point_2 = keypoints_2[good_matches[i].trainIdx].pt;

        std::cout << point_1.x << " " << point_1.y << std::endl;
        std::cout << point_2.x << " " << point_2.y << std::endl;

        points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    // the following does not contribute to the end result
    Mat h = findHomography(points1, points2, RANSAC);
    Mat img1_reg_result;
    warpPerspective(input_1, img1_reg_result, h, input_2.size());
    imwrite("reg.jpg", img1_reg_result);
    imwrite("dst.jpg", input_2);

    return std::make_pair(points1, points2);
}

std::tuple<int, int, int> rgb_texture_new(rs2::video_frame texture, rs2::texture_coordinate texture_coords) {
    // Get Width and Height coordinates of texture
    int width  = texture.get_width();  // Frame width in pixels
    int height = texture.get_height(); // Frame height in pixels

    // Normals to Texture Coordinates conversion
    int x_value = std::min(std::max(int(texture_coords.u * width + .5f), 0), width - 1);
    int y_value = std::min(std::max(int(texture_coords.v * height + .5f), 0), height - 1);

    int bytes = x_value * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
    int strides = y_value * texture.get_stride_in_bytes(); // Get line width in bytes
    int text_index = (bytes + strides);

    const auto new_texture = reinterpret_cast<const uint8_t*>(texture.get_data());

    // RGB components to save in tuple
    int nt1 = new_texture[text_index];
    int nt2 = new_texture[text_index + 1];
    int nt3 = new_texture[text_index + 2];

    return { nt1, nt2, nt3 };
}

rgb_point_cloud_pointer convert_to_pcl_new(const rs2::points& points, const rs2::video_frame& color) {
    rgb_point_cloud_pointer cloud(new rgb_point_cloud);

    std::tuple<uint8_t, uint8_t, uint8_t> _rgb_texture;

    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width  = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize((int)points.size());

    auto texture_coordinates = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < (int)points.size(); i++) {
        cloud->points[i].x = vertices[i].x;
        cloud->points[i].y = vertices[i].y;
        cloud->points[i].z = vertices[i].z;

        // Obtain color texture for specific point
        _rgb_texture = rgb_texture(color, texture_coordinates[i]);

        // Mapping Color (BGR due to Camera Model)
        cloud->points[i].r = get<2>(_rgb_texture); // Reference tuple<2>
        cloud->points[i].g = get<1>(_rgb_texture); // Reference tuple<1>
        cloud->points[i].b = get<0>(_rgb_texture); // Reference tuple<0>
    }

    return cloud;
}

/*

rgb_point_cloud_pointer convert_to_pcl_new(const rs2::points& points, const rs2::video_frame& color) {
    rgb_point_cloud_pointer cloud(new rgb_point_cloud);

    std::tuple<uint8_t, uint8_t, uint8_t> _rgb_texture;

    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width = static_cast<uint32_t>(sp.width() * 3 / 5);
    cloud->height = static_cast<uint32_t>(sp.height() * 3 / 5);
    cloud->points.resize(cloud->width * cloud->height);

    auto texture_coordinates = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    int i = 0;
    for (int r = sp.height() / 5; r < sp.height() / 5 * 4; r++){
        for (int c = sp.width() / 5; c < sp.width() / 5 * 4; c++){
            int vertices_index = r * (sp.width()) + c;

            cloud->points[i].x = vertices[vertices_index].x;
            cloud->points[i].y = vertices[vertices_index].y;
            cloud->points[i].z = vertices[vertices_index].z;

            // Obtain color texture for specific point
            _rgb_texture = rgb_texture_new(color, texture_coordinates[vertices_index]);

            // Mapping Color (BGR due to Camera Model)
            cloud->points[i].r = get<2>(_rgb_texture); // Reference tuple<2>
            cloud->points[i].g = get<1>(_rgb_texture); // Reference tuple<1>
            cloud->points[i].b = get<0>(_rgb_texture); // Reference tuple<0>
            i++;
        }
    }

    return cloud;
}

*/

/*
 * Filter using two mechanisms
 */
rgb_point_cloud_pointer filter_pcl_new(rgb_point_cloud_pointer cloud) {
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    rgb_point_cloud_pointer cloud_pass_through(new rgb_point_cloud);
    rgb_point_cloud_pointer cloud_sor(new rgb_point_cloud);
    rgb_point_cloud_pointer cloud_voxel_grid(new rgb_point_cloud);

    // 1. Applies pass through filter
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.filter(*cloud_pass_through);
    pass.setFilterLimits(0.2, 2.5);

    // 2. Applies sor filter
    pcl::StatisticalOutlierRemoval<rgb_point> sor;
    sor.setInputCloud(cloud_pass_through);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.5);
    sor.filter(*cloud_sor);

    return cloud_voxel_grid;
}

std::pair<std::vector<rgb_point_cloud_pointer>, std::vector<float3>> get_clouds_new(rs2::pipeline pipe, int nr_frames) {
    std::vector<rgb_point_cloud_pointer> clouds;
    std::vector<rs2::frameset> framesets;
    std::vector<rs2::video_frame> color_frames;
    std::vector<rs2::video_frame> depth_frames;
    std::vector<float3> thetas;

    rs2::pointcloud pc;
    rs2::points points;

    std::mutex mutex;

    RotationEstimator algo;

    int frame = 0;

    auto time = std::chrono::system_clock::now();

    std::cout << "[RS]  Starting capture sequence..." << std::endl;
    while (frame < nr_frames) {
        auto frameset = pipe.wait_for_frames();

        auto gyro_frame = frameset.first_or_default(RS2_STREAM_GYRO);
        auto gyro_motion = gyro_frame.as<rs2::motion_frame>();

        auto accel_frame = frameset.first_or_default(RS2_STREAM_ACCEL);
        auto accel_motion = accel_frame.as<rs2::motion_frame>();

        rs2_vector gyro_data = gyro_motion.get_motion_data();
        algo.process_gyro(gyro_data, gyro_motion.get_timestamp());
        rs2_vector accel_data = accel_motion.get_motion_data();
        algo.process_accel(accel_data);

        // Get computed rotation
        float3 theta = algo.get_theta();

        auto now = std::chrono::system_clock::now();
        if ((now - time).count() < 2000000000)
            continue;

        time = now;

        std::cout << "[RS]    Captured frame [" << frame << "]" << std::endl;
        framesets.push_back(frameset);
        thetas.push_back(theta);
        ++frame;
    }

    std::cout << "[RS]  Converting framesets to point clouds..." << std::endl << std::flush;

    for (auto frameset: framesets) {
        auto color = frameset.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frameset.get_infrared_frame();

        color_frames.push_back(color);

        pc.map_to(color);

        auto depth = frameset.get_depth_frame();
        depth_frames.push_back(depth);

        points = pc.calculate(depth);
        auto pcl = convert_to_pcl_new(points, color);
        clouds.push_back(pcl);
    }

    for (int i = 0; i < (int)framesets.size() - 1; i++) {
        auto color_0 = color_frames[i];
        auto color_1 = color_frames[i + 1];

        auto corresponding_points = get_keypoints_twoframes(color_0, color_1);
        auto points_0 = corresponding_points.first;
        auto points_1 = corresponding_points.second;

        auto cloud_0 = clouds[i];
        auto cloud_1 = clouds[i + 1];

        std::vector<rgb_point> cloud_points_0;
        std::vector<rgb_point> cloud_points_1;

        for (size_t pi = 0; pi < points_0.size(); pi++) {
            auto point_0 = points_0[pi];
            auto point_1 = points_1[pi];

            auto cloud_point_0 = cloud_0->at((int)point_0.x, (int)point_0.y);
            auto cloud_point_1 = cloud_1->at((int)point_1.x, (int)point_1.y);

            cloud_points_0.push_back(cloud_point_0);
            cloud_points_1.push_back(cloud_point_1);
        }
    }

    waitKey(0);

    return std::make_pair(clouds, thetas);
}
