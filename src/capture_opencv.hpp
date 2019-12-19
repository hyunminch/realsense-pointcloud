#include <librealsense2/rs.hpp>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/transformation_estimation_dual_quaternion.h>
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
#include "translation_estimator.hpp"
#include "blur_filter.hpp"
#include "file_name_format.hpp"
#include "edge_extractor.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

struct RelativeCameraPose {
    float3 theta;
    Eigen::Translation3f translation;
};

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

std::pair<std::vector<Point2f>, std::vector<Point2f>> get_keypoints_twoframes(rs2::video_frame color_frame1, rs2::video_frame color_frame2, const std::string prefix, int frame_no) {
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

    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    std::vector<DMatch> best_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    std::sort(good_matches.begin(), good_matches.end());
    good_matches.erase(good_matches.begin () + (int)good_matches.size() / 2, good_matches.end());

    // Extract location of good matches
    std::vector<Point2f> points1, points2;
    std::vector<Point2f> rpoints1, rpoints2;

    for (size_t i = 0; i < good_matches.size(); i++) {
        auto point_1 = keypoints_1[good_matches[i].queryIdx].pt;
        auto point_2 = keypoints_2[good_matches[i].trainIdx].pt;

        points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    std::vector<Point2f> cmp;
    // the following does not contribute to the end result
    Mat h = findHomography(points1, points2, RANSAC);

    perspectiveTransform(points1, cmp, h);

    for (int i = 0; i < (int)points1.size(); i++) {
        auto c = cmp[i];
        auto p2 = points2[i];

        float dx = p2.x - c.x;
        float dy = p2.y - c.y;

        float d = dx * dx + dy * dy;

        if (d <= 10) {
            std::cout << "C: " << c << std::endl;
            std::cout << "P2: " << p2 << std::endl;

            rpoints1.push_back(points1[i]);
            rpoints2.push_back(points2[i]);
            best_matches.push_back(good_matches[i]);
        } else {
            std::cout << "Rejected!" << std::endl;
        }

    }
//    imwrite("reg.jpg", img1_reg_result);
//    imwrite("dst.jpg", input_2);

    // Draw top matches
    Mat im_matches;
    drawMatches(input_1, keypoints_1, input_2, keypoints_2, best_matches, im_matches);
    imwrite(fn_matches(prefix, frame_no), im_matches);
//    imwrite("dataset/" + prefix + "-matches-" + std::to_string(frame_no) + ".jpg", im_matches);

    return std::make_pair(rpoints1, rpoints2);
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

        if (cloud->points[i].z < 0.2 || cloud->points[i].z > 1.5)
            cloud->points[i].z = 0.0;

        // Obtain color texture for specific point
        _rgb_texture = rgb_texture(color, texture_coordinates[i]);

        // Mapping Color (BGR due to Camera Model)
        cloud->points[i].r = get<0>(_rgb_texture); // Reference tuple<2>
        cloud->points[i].g = get<1>(_rgb_texture); // Reference tuple<1>
        cloud->points[i].b = get<2>(_rgb_texture); // Reference tuple<0>
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
    pass.setFilterLimits(0.2, 2.5);
    pass.filter(*cloud_pass_through);

    // 2. Applies sor filter
//    pcl::StatisticalOutlierRemoval<rgb_point> sor;
//    sor.setInputCloud(cloud_pass_through);
//    sor.setMeanK(50);
//    sor.setStddevMulThresh(1.5);
//    sor.filter(*cloud_sor);

    return cloud_pass_through;
}

void make_thetas_relative(std::vector<float3>& thetas) {
//    float3 init_theta = thetas[0] * -1.0;
//    for (int i = 0; i < thetas.size(); i++)
//        thetas[i].add(init_theta.x, init_theta.y, init_theta.z);

    for (int i = (int)thetas.size() - 1; i >= 0; i--) {
        float3 prev_theta = thetas[i - 1] * -1.0;
        thetas[i].add(prev_theta.x, prev_theta.y, prev_theta.z);
    }
}

rgb_point_cloud_pointer get_clouds_new(rs2::pipeline pipe, int nr_frames, const std::string prefix="test") {
//std::vector<std::pair<rgb_point_cloud_pointer, Eigen::Matrix4f>> get_clouds_new(rs2::pipeline pipe, int nr_frames) {
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
        if ((now - time).count() < 6000000000)
            continue;

        time = now;

        std::cout << "[RS]    Captured frame [" << frame << "]" << std::endl;
        framesets.push_back(frameset);
        thetas.push_back(theta);
        ++frame;
    }

    std::cout << "[RS]  Converting framesets to point clouds..." << std::endl << std::flush;

    rs2::spatial_filter spatial_filter;

    spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.80f);

    for (int i = 0; i < framesets.size(); i++) {
        auto color = framesets[i].get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = framesets[i].get_infrared_frame();

        color_frames.push_back(color);

        pc.map_to(color);

        auto depth = framesets[i].get_depth_frame();
        auto filtered_depth = spatial_filter.process(depth);
        depth_frames.push_back(filtered_depth);

        points = pc.calculate(filtered_depth);
        auto pcl = convert_to_pcl_new(points, color);
//        auto filtered = filter_pcl_new(pcl);
//        pcl::io::savePCDFile("dataset/" + prefix + "-raw-" + std::to_string(i) + ".pcd", *pcl);
        BlurFilter blur_filter;
        blur_filter.filter(pcl);
        pcl::io::savePCDFile(fn_raw(prefix, i), *pcl);
        clouds.push_back(pcl);
    }

    make_thetas_relative(thetas);

    assert(thetas.size() == clouds.size());
    std::vector<std::pair<rgb_point_cloud_pointer, Eigen::Matrix4f>> result;
    // No transformation for the first cloud.
    Eigen::Matrix4f init_transformation = (Eigen::Translation3f(0, 0, 0) * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())).matrix();
    result.push_back(std::make_pair(clouds[0], init_transformation));
    TranslationEstimator translation_estimator;

    for (int i = 1; i < (int)framesets.size(); i++) {
        auto color_0 = color_frames[i - 1];
        auto color_1 = color_frames[i];

        auto corresponding_points = get_keypoints_twoframes(color_0, color_1, prefix, i);
        auto points_0 = corresponding_points.first;
        auto points_1 = corresponding_points.second;

        auto cloud_0 = clouds[i - 1];
        auto cloud_1 = clouds[i];

        std::vector<std::pair<rgb_point, rgb_point>> kpt_correspondences;

        for (size_t pi = 0; pi < points_0.size(); pi++) {
            auto point_0 = points_0[pi];
            auto point_1 = points_1[pi];

            auto cloud_point_0 = cloud_0->at((int)point_0.x, (int)point_0.y);
            auto cloud_point_1 = cloud_1->at((int)point_1.x, (int)point_1.y);

            if (!(cloud_point_0.z < 0.01 || cloud_point_1.z < 0.01)) {
                kpt_correspondences.push_back(std::make_pair(cloud_point_0, cloud_point_1));
                std::cout << cloud_point_0 << " " << cloud_point_1 << std::endl;
            }
        }

        rgb_point_cloud_pointer in(new rgb_point_cloud);
        rgb_point_cloud_pointer out(new rgb_point_cloud);

//        int cor_size = 3;
        int cor_size = (int)kpt_correspondences.size();

//        in->width = (int)kpt_correspondences.size();
        in->width = cor_size;
        in->height = 1;
        in->is_dense = false;
        in->resize(in->width * in->height);

//        out->width = (int)kpt_correspondences.size();
        out->width = cor_size;
        out->height = 1;
        out->is_dense = false;
        out->resize(out->width * out->height);

//        for (size_t pi = 0; pi < kpt_correspondences.size(); pi++) {
        for (size_t pi = 0; pi < cor_size; pi++) {

            auto pair = kpt_correspondences[pi];
//            auto point_0 = points_0[pi];
//            auto point_1 = points_1[pi];
            auto cloud_point_0 = pair.first;
            auto cloud_point_1 = pair.second;

            in->points[pi].x = cloud_point_0.x;
            in->points[pi].y = cloud_point_0.y;
            in->points[pi].z = cloud_point_0.z;

            out->points[pi].x = cloud_point_1.x;
            out->points[pi].y = cloud_point_1.y;
            out->points[pi].z = cloud_point_1.z;
        }

//        pcl::io::savePCDFile("dataset/" + prefix + "-cor-kpts-" + std::to_string(i) + "-in.pcd", *in);
//        pcl::io::savePCDFile("dataset/" + prefix + "-cor-kpts-" + std::to_string(i) + "-out.pcd", *out);
        pcl::io::savePCDFile(fn_cor_kpts(prefix, i - 1, i), *in);
        pcl::io::savePCDFile(fn_cor_kpts(prefix, i, i - 1), *out);

//        pcl::registration::TransformationEstimation3Point<pcl::PointXYZRGB, pcl::PointXYZRGB> threepoint;
//        Eigen::Matrix4f transformation2;
//        threepoint.estimateRigidTransformation(*out, *in, transformation2);

//        pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB,pcl::PointXYZRGB> TESVD;
//        pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB,pcl::PointXYZRGB>::Matrix4 transformation2;
//        TESVD.estimateRigidTransformation (*out,*in,transformation2);

//        pcl::registration::TransformationEstimationLM<pcl::PointXYZRGB, pcl::PointXYZRGB> lm;
//        pcl::registration::TransformationEstimationLM<pcl::PointXYZRGB,pcl::PointXYZRGB>::Matrix4 transformation2;
//        lm.estimateRigidTransformation (*out,*in,transformation2);

        pcl::registration::TransformationEstimationDualQuaternion<pcl::PointXYZRGB, pcl::PointXYZRGB> dq;
        pcl::registration::TransformationEstimationDualQuaternion<pcl::PointXYZRGB,pcl::PointXYZRGB>::Matrix4 transformation2;
        dq.estimateRigidTransformation (*out,*in,transformation2);
        std::ofstream fstream;
//        fstream.open("dataset/" + prefix + "-transmat-" + std::to_string(i) + "-to-" + std::to_string(i - 1) + ".txt");
        fstream.open(fn_transmat(prefix, i, i - 1));
        fstream << transformation2 << std::endl;
        fstream.close();

        for (size_t pi = 0; pi < cor_size; pi++) {
            auto pair = kpt_correspondences[pi];
            auto cloud_point_0 = pair.first;
            auto cloud_point_1 = pair.second;

            Eigen::Vector4f p1(cloud_point_1.x, cloud_point_1.y, cloud_point_1.z, 1);
            Eigen::Vector4f mult = transformation2 * p1;
            std::cout << "Mult: " << mult << std::endl;
            std::cout << "P0: " << cloud_point_0 << std::endl;
        }

        Eigen::Translation3f translation = translation_estimator.estimate_translation(kpt_correspondences, thetas[i]);
        Eigen::AngleAxisf rotation_x(thetas[i].z, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf rotation_y(-thetas[i].y, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf rotation_z(-thetas[i].x, Eigen::Vector3f::UnitZ());
        Eigen::Matrix4f transformation = (translation * rotation_x * rotation_y * rotation_z).matrix();

        std::cout << "Rotation X: " << rotation_x.angle() << std::endl;
        std::cout << "Rotation Y: " << rotation_y.angle() << std::endl;
        std::cout << "Rotation Z: " << rotation_z.angle() << std::endl;
        std::cout << "Transformation: " << transformation << std::endl;
        std::cout << "Translation: " << translation.x() << " " << translation.y() << " " << translation.z() << std::endl;

        result.push_back(std::make_pair(clouds[i], transformation2));
    }

    pcl::ApproximateVoxelGrid<rgb_point> approx_voxel_grid;
    pcl::IterativeClosestPoint<rgb_point, rgb_point> icp;
    pcl::registration::CorrespondenceRejectorTrimmed::Ptr cor_rej_trimmed(new pcl::registration::CorrespondenceRejectorTrimmed);

    int target_idx = 0;
//    BlurFilter blur_filter;
//
//    for (int cloud_idx = 0; cloud_idx < (int)clouds.size(); cloud_idx++) {
//        blur_filter.filter(clouds[cloud_idx]);
//    }

    rgb_point_cloud_pointer target_cloud = extract_edge_features(clouds[0]);
    rgb_point_cloud_pointer global_cloud = clouds[0];

    // these cloud pointers are to be used as temporary variables
    rgb_point_cloud_pointer downsized_src(new rgb_point_cloud);
    rgb_point_cloud_pointer downsized_dst(new rgb_point_cloud);

    approx_voxel_grid.setLeafSize(0.005, 0.005, 0.005);

    icp.setMaximumIterations(300);
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setTransformationEpsilon(0.5);
    icp.setEuclideanFitnessEpsilon(0.25);
    icp.setRANSACOutlierRejectionThreshold(0.05);

    Eigen::Matrix4f init_guess = (Eigen::Translation3f(0, 0, 0) * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())).matrix();
    std::cout << init_guess << std::endl;

//    pcl::io::savePCDFile(fn_no_blur(prefix, target_idx), *target_cloud);
//    pcl::io::savePCDFile("dataset/" + prefix + "-no-blur-" + std::to_string(target_idx) + ".pcd", *target_cloud);

    for (int cloud_idx = 1; cloud_idx < (int)clouds.size(); cloud_idx++) {
        rgb_point_cloud_pointer aligned(new rgb_point_cloud);
        rgb_point_cloud_pointer original_transformed(new rgb_point_cloud);
        rgb_point_cloud_pointer temp(new rgb_point_cloud);

        init_guess = result[cloud_idx].second * init_guess;
        std::cout << init_guess << std::endl;
        std::ofstream fstream;
//        fstream.open("dataset/" + prefix + "-guess-" + std::to_string(cloud_idx) + "-to-" + std::to_string(target_idx) + ".txt");
        fstream.open(fn_guess(prefix, cloud_idx, target_idx));
        fstream << init_guess << std::endl;
        fstream.close();

//        pcl::io::savePCDFile(fn_no_blur(prefix, cloud_idx), *clouds[cloud_idx]);
//        pcl::io::savePCDFile("dataset/" + prefix + "-no-blur-" + std::to_string(cloud_idx) + ".pcd", *target_cloud);
//        pcl::transformPointCloud(*clouds[cloud_idx], *temp, init_guess);

        auto input_edges = extract_edge_features(clouds[cloud_idx]);
        pcl::transformPointCloud(*input_edges, *input_edges, init_guess);
        pcl::transformPointCloud(*clouds[cloud_idx], *original_transformed, init_guess);

        icp.setInputSource(original_transformed);
        icp.setInputTarget(global_cloud);
        icp.align(*aligned);

        if (icp.hasConverged()) {
            rgb_point_cloud_pointer transformed(new rgb_point_cloud);
            pcl::transformPointCloud(*input_edges, *transformed, icp.getFinalTransformation());
            pcl::transformPointCloud(*original_transformed, *temp, icp.getFinalTransformation());
            *target_cloud += *transformed;
            pcl::io::savePCDFile(fn_result_upto(prefix, cloud_idx), *target_cloud);
            *global_cloud += *temp;
        }
    }

    return global_cloud;
}
