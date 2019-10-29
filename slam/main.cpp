#include <librealsense2/rs.hpp>
#include "utils.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct state {
    state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0),
              ml(false), offset_x(0.0f), offset_y(0.0f) {}
    double yaw, pitch, last_x, last_y; bool ml; float offset_x, offset_y;
};

typedef pcl::PointXYZRGB rgb_cloud;
typedef pcl::PointCloud<rgb_cloud> point_cloud;
typedef point_cloud::Ptr cloud_pointer;

using pcl_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points);

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

cloud_pointer convert_to_pcl(const rs2::points& points, const rs2::video_frame& color) {
    cloud_pointer cloud(new point_cloud);

    std::tuple<uint8_t, uint8_t, uint8_t> _rgb_texture;

    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width  = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize(points.size());

    auto texture_coordinates = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < (int)points.size(); i++)
    {
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

    return cloud; // PCL RGB Point Cloud generated
}

std::vector<cloud_pointer> get_clouds(rs2::pipeline pipe, int nr_frames) {
    int _nr_frames = nr_frames;

    std::vector<cloud_pointer> clouds;

    rs2::pointcloud pc;
    rs2::points points;

    while (_nr_frames--) {
        std::cout << "Capturing..." << std::endl;

        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        auto color = frames.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frames.get_infrared_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);

        auto pcl = convert_to_pcl(points, color);
        clouds.push_back(pcl);
    }

    return clouds;
}

void register_glfw_callbacks(window& app, state& app_state);

int main(int argc, char * argv[]) try {
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    auto clouds = get_clouds(pipe, 2);

    cloud_pointer registration_result(new point_cloud);
    
    // Below is the ICP implementation.
    /*
    pcl::IterativeClosestPoint<rgb_cloud, rgb_cloud> icp;

    icp.setMaximumIterations(5);
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1);
    icp.setInputSource(clouds[0]);
    icp.setInputTarget(clouds[1]);
    icp.align(*registration_result);

    if (icp.hasConverged()) {
        std::cout << "Converged!" << std::endl;

        // Create a simple OpenGL window for rendering:
        window app(1280, 720, "RealSense PCL Pointcloud Example");
        // Construct an object to manage view state
        state app_state;
        // register callbacks to allow manipulation of the pointcloud
        register_glfw_callbacks(app, app_state);

        while (app) {
            draw_pointcloud(app, app_state, {registration_result});
        }
    } 
    */
    
    // Below is the NDT implementation.
    pcl::NormalDistributionsTransform<rgb_cloud, rgb_cloud> ndt;

    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setMaximumIterations(35);
    ndt.setInputSource(clouds[0]);
    ndt.setInputTarget(clouds[1]);

    Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
    Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

    ndt.align(*registration_result, init_guess); 
    
    if (ndt.hasConverged()) {
        std::cout << "NDT has converged!" << std::endl;
        
        // Create a simple OpenGL window for rendering:
        window app(1280, 720, "RealSense PCL Pointcloud Example");
        // Construct an object to manage view state
        state app_state;
        // register callbacks to allow manipulation of the pointcloud
        register_glfw_callbacks(app, app_state);

        while (app) {
            draw_pointcloud(app, app_state, {registration_result});
        }
    } else
        std::cout << "Has not converged!" << std::endl;

    return EXIT_SUCCESS;
} catch (const rs2::error & e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

// Registers the state variable and callbacks to allow mouse control of the pointcloud
void register_glfw_callbacks(window& app, state& app_state) {
    app.on_left_mouse = [&](bool pressed) {
        app_state.ml = pressed;
    };

    app.on_mouse_scroll = [&](double xoffset, double yoffset) {
        app_state.offset_x += static_cast<float>(xoffset);
        app_state.offset_y += static_cast<float>(yoffset);
    };

    app.on_mouse_move = [&](double x, double y) {
        if (app_state.ml) {
            app_state.yaw -= (x - app_state.last_x);
            app_state.yaw = std::max(app_state.yaw, -120.0);
            app_state.yaw = std::min(app_state.yaw, +120.0);
            app_state.pitch += (y - app_state.last_y);
            app_state.pitch = std::max(app_state.pitch, -80.0);
            app_state.pitch = std::min(app_state.pitch, +80.0);
        }
        app_state.last_x = x;
        app_state.last_y = y;
    };

    app.on_key_release = [&](int key) {
        // Escape
        if (key == 32) {
            app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
        }
    };
}

void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points) {
    // OpenGL commands that prep screen for the pointcloud
    glPopMatrix();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    float width = app.width(), height = app.height();

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y*0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glPointSize(width / 640);
    glEnable(GL_TEXTURE_2D);

    int color = 0;

    for (auto&& pc : points) {
        glBegin(GL_POINTS);

        /* this segment actually prints the pointcloud */
        for (auto & p : pc->points) {
            if (p.z) {
                // upload the point and texture coordinates only for points we have depth data for
                glColor3f(p.r / 255.0, p.g / 255.0, p.b / 255.0);
                glVertex3f(p.x, p.y, p.z);
            }
        }

        glEnd();
    }

    // OpenGL cleanup
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
    glPushMatrix();
}
