#include <librealsense2/rs.hpp>
#include <mutex>
#include <cstring>
#include "capture.hpp"

class rotation_estimator
{
    // theta is the angle of camera rotation in x, y and z components
    float3 theta;
    std::mutex theta_mtx;
    /* alpha indicates the part that gyro and accelerometer take in computation of theta; higher alpha gives more weight to gyro, but too high
    values cause drift; lower alpha gives more weight to accelerometer, which is more sensitive to disturbances */
    float alpha = 0.98;
    bool first = true;
    // Keeps the arrival time of previous gyro frame
    double last_ts_gyro = 0;
public:
    // Function to calculate the change in angle of motion based on data from gyro
    void process_gyro(rs2_vector gyro_data, double ts)
    {
        if (first) // On the first iteration, use only data from accelerometer to set the camera's initial position
        {
            last_ts_gyro = ts;
            return;
        }
        // Holds the change in angle, as calculated from gyro
        float3 gyro_angle;

        // Initialize gyro_angle with data from gyro
        gyro_angle.x = gyro_data.x; // Pitch
        gyro_angle.y = gyro_data.y; // Yaw
        gyro_angle.z = gyro_data.z; // Roll

        // Compute the difference between arrival times of previous and current gyro frames
        double dt_gyro = (ts - last_ts_gyro) / 1000.0;
        last_ts_gyro = ts;

        // Change in angle equals gyro measures * time passed since last measurement
        gyro_angle = gyro_angle * dt_gyro;

        // Apply the calculated change of angle to the current angle (theta)
        std::lock_guard<std::mutex> lock(theta_mtx);
        theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x);
    }

    void process_accel(rs2_vector accel_data)
    {
        // Holds the angle as calculated from accelerometer data
        float3 accel_angle;

        // Calculate rotation angle from accelerometer data
        accel_angle.z = atan2(accel_data.y, accel_data.z);
        accel_angle.x = atan2(accel_data.x, sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z));

        // If it is the first iteration, set initial pose of camera according to accelerometer data (note the different handling for Y axis)
        std::lock_guard<std::mutex> lock(theta_mtx);
        if (first)
        {
            first = false;
            theta = accel_angle;
            // Since we can't infer the angle around Y axis using accelerometer data, we'll use PI as a convetion for the initial pose
            theta.y = PI;
        }
        else
        {
            /*
            Apply Complementary Filter:
                - high-pass filter = theta * alpha:  allows short-duration signals to pass through while filtering out signals
                  that are steady over time, is used to cancel out drift.
                - low-pass filter = accel * (1- alpha): lets through long term changes, filtering out short term fluctuations
            */
            theta.x = theta.x * alpha + accel_angle.x * (1 - alpha);
            theta.z = theta.z * alpha + accel_angle.z * (1 - alpha);
        }
    }

    // Returns the current rotation angle
    float3 get_theta()
    {
        std::lock_guard<std::mutex> lock(theta_mtx);
        return theta;
    }
};


std::vector<rgb_point_cloud_pointer> get_clouds_camera_motion(rs2::pipeline pipe, int nr_frames){
	std::vector<rgb_point_cloud_pointer>	clouds;
	rs2::pointcloud pc;
	rs2::points points;

	rs2::config config;

	config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
	config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
	config.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 15);
	config.enable_stream(RS2_STREAM_COLOR,1280, 720, RS2_FORMAT_BGR8, 15);
	config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 15);
	rotation_estimator camera_rotate;
	pipe.start(config);
		for (int f = 0; f < nr_frames; f++){
			auto frames = pipe.wait_for_frames();
			auto camera_accel = frames.first(RS2_STREAM_ACCEL).as<rs2::motion_frame>();
			auto camera_gyro = frames.first(RS2_STREAM_GYRO).as<rs2::motion_frame>();
			auto accel_data = camera_accel.get_motion_data();
			cout << "Camera Accel data : "<<accel_data<<endl;
			auto gyro_data = camera_gyro.get_motion_data();
			cout << "Camera gyro data : "<<gyro_data<<endl;
			auto color = frames.get_color_frame();
			cout << "[RS] Capture start"<<endl;
			if (!color) color = frames.get_infrared_frame();
			pc.map_to(color);
			auto depth = frames.get_depth_frame();
			points = pc.calculate(depth);

			auto pcl = convert_to_pcl(points, color);
			auto filtered = filter_pcl(pcl);
			std::cout<<"  [RS] Successfully filtered" << std::endl;
			clouds.push_back(filtered);
			std::cout << "[RS] Captured frame[" << f<<"]"<<std::endl;
			sleep(2);
		}	
	pipe.stop();
	return clouds;
}
