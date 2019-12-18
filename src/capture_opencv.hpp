#include <librealsense2/rs.hpp>
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

void get_keypoints_twoframes(rs2::video_frame color_frame1, rs2::video_frame color_frame2) {
    Mat input1(Size(1280, 720), CV_8UC3, (void*)color_frame1.get_data(), Mat::AUTO_STEP);
    Mat input2(Size(1280, 720), CV_8UC3, (void*)color_frame2.get_data(), Mat::AUTO_STEP);

    auto result1 = get_keypoints(0, input1);
    auto result2 = get_keypoints(1, input2);

    auto keypoints1 = result1.first;
    auto keypoints2 = result2.first;
    auto descriptors1 = result1.second;
    auto descriptors2 = result2.second;

    // Match features.
//    std::vector<DMatch> matches;
//    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
//    matcher->match(keypoints1, keypoints2, matches, Mat());

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    std::sort(good_matches.begin(), good_matches.end());
    good_matches.erase(good_matches.begin() + good_matches.size() / 2, good_matches.end());

    // Draw top matches
    Mat imMatches;
    drawMatches(input1, keypoints1, input2, keypoints2, good_matches, imMatches);
    imwrite("matches.jpg", imMatches);

    // Extract location of good matches
    std::vector<Point2f> points1, points2;

//    for (size_t i = 0; i < matches.size(); i++) {
//        points1.push_back(keypoints1[matches[i].data()->queryIdx].pt);
//        points2.push_back(keypoints2[matches[i].data()->trainIdx].pt);
//    }
}

std::tuple<int, int, int> rgb_texture(rs2::video_frame texture, rs2::texture_coordinate texture_coords) {
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

rgb_point_cloud_pointer convert_to_pcl(const rs2::points& points, const rs2::video_frame& color) {
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
            _rgb_texture = rgb_texture(color, texture_coordinates[vertices_index]);

            // Mapping Color (BGR due to Camera Model)
            cloud->points[i].r = get<2>(_rgb_texture); // Reference tuple<2>
            cloud->points[i].g = get<1>(_rgb_texture); // Reference tuple<1>
            cloud->points[i].b = get<0>(_rgb_texture); // Reference tuple<0>
            i++;
        }
    }

    return cloud;
    cloud->is_dense = true;
}

/*
 * Filter using two mechanisms
 */
rgb_point_cloud_pointer filter_pcl(rgb_point_cloud_pointer cloud) {
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

std::pair<std::vector<rgb_point_cloud_pointer>, std::vector<float3>> get_clouds(rs2::pipeline pipe, int nr_frames) {
    std::vector<rgb_point_cloud_pointer> clouds;
    std::vector<rs2::frameset> framesets;
    std::vector<rs2::video_frame> color_frames;
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

    frame = 0;
    for (auto frameset: framesets) {
        auto color = frameset.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frameset.get_infrared_frame();

//        get_keypoints(frame++, color);
        color_frames.push_back(color);

        pc.map_to(color);

        auto depth = frameset.get_depth_frame();
        points = pc.calculate(depth);
        auto pcl = convert_to_pcl(points, color);
        clouds.push_back(pcl);
    }

    get_keypoints_twoframes(color_frames[0], color_frames[1]);

    waitKey(0);

    return std::make_pair(clouds, thetas);
}
