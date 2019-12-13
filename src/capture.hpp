#include <librealsense2/rs.hpp>
#include <vector>
#include <chrono>
#include <thread>

#include "types.hpp"
#include "utils.hpp"

#include "rotation_estimator.hpp"

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

/*
rgb_point_cloud_pointer convert_to_pcl(const rs2::points& points, const rs2::video_frame& color) {
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
*/

rgb_point_cloud_pointer convert_to_pcl(const rs2::points& points, const rs2::video_frame& color) {
    rgb_point_cloud_pointer cloud(new rgb_point_cloud);

    std::tuple<uint8_t, uint8_t, uint8_t> _rgb_texture;

    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width = static_cast<uint32_t>(sp.width()* 3 / 5);
    cloud->height = static_cast<uint32_t>(sp.height() * 3 / 5);
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    auto texture_coordinates = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    int i = 0;
    for (int r = sp.width() / 5; r < sp.width() / 5 * 4; r++){
        for (int c = sp.height() / 5; c < sp.height() / 5 * 4; c++){
            int vertices_index = c * (sp.width()) + r;

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

        pc.map_to(color);

        auto depth = frameset.get_depth_frame();
        points = pc.calculate(depth);
        auto pcl = convert_to_pcl(points, color);
        clouds.push_back(pcl);
    }

    return std::make_pair(clouds, thetas);
}
