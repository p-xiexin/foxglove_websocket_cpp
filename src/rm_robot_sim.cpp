#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>

#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/util/time_util.h>

#include <foxglove/websocket/base64.hpp>
#include <foxglove/websocket/server_factory.hpp>
#include <foxglove/websocket/websocket_notls.hpp>
#include <foxglove/websocket/websocket_server.hpp>

#include <opencv2/opencv.hpp>

#include "foxglove/SceneUpdate.pb.h"
#include "foxglove/FrameTransforms.pb.h"
#include "foxglove/CompressedImage.pb.h"
#include "foxglove/CameraCalibration.pb.h"
#include "robot/robot_armor.hpp"
#include "robot/camera.hpp"

std::atomic<bool> running(true);

// 获取当前时间的纳秒数
static uint64_t nanosecondsSinceEpoch()
{
	return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
						std::chrono::system_clock::now().time_since_epoch())
						.count());
}

// 将图像转换为 CompressedImage 消息
foxglove::CompressedImage createCompressedImageMessage(const cv::Mat& image, const std::string& format, const std::string& frame_id) {
    // 将图像编码为字节流
    std::vector<uchar> buf;
    cv::imencode(".png", image, buf);
    std::string imageBytes(reinterpret_cast<char*>(buf.data()), buf.size());

    // 创建 CompressedImage 消息并设置相关参数
    foxglove::CompressedImage img_msg;
    img_msg.set_format(format);
    img_msg.set_frame_id(frame_id);
    img_msg.set_data(imageBytes);

    return img_msg;
}

foxglove::CameraCalibration createCameraCalibration(const Camera& camera, int image_width, int image_height, const std::string& distortion_model, const std::string& frame_id, uint16_t now) {
    // 创建一个 CameraCalibration 实例
    foxglove::CameraCalibration calibration;

    // 设置图像宽度和高度
    calibration.set_width(image_width);
    calibration.set_height(image_height);

    // 设置失真模型
    calibration.set_distortion_model(distortion_model);

    // 设置失真参数 D
    std::vector<double> distortion = camera.getDistortionParams();
    for (auto d : distortion) {
        calibration.add_d(d);
    }

    // 设置固有相机矩阵 K
    Eigen::Matrix3d intrinsic_matrix = camera.getIntrinsicMatrix();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            calibration.add_k(intrinsic_matrix(i, j));
        }
    }

    // 设置整流矩阵 R（单位矩阵）
    Eigen::Matrix3d rectification_matrix = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            calibration.add_r(rectification_matrix(i, j));
        }
    }

    // 计算投影矩阵 P
    Eigen::Matrix<double, 3, 4> projection_matrix = Eigen::Matrix<double, 3, 4>::Zero();
    projection_matrix.block<3,3>(0,0) = intrinsic_matrix;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            calibration.add_p(projection_matrix(i, j));
        }
    }

    // 设置时间戳
    *calibration.mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now);

    // 设置帧ID
    calibration.set_frame_id(frame_id);

    return calibration;
}

// 将指定描述符及其所有依赖项序列化为字符串，用作通道模式
static std::string SerializeFdSet(const google::protobuf::Descriptor *toplevelDescriptor)
{
	google::protobuf::FileDescriptorSet fdSet;
	std::queue<const google::protobuf::FileDescriptor *> toAdd;
	toAdd.push(toplevelDescriptor->file());
	std::unordered_set<std::string> seenDependencies;
	while (!toAdd.empty())
	{
		const google::protobuf::FileDescriptor *next = toAdd.front();
		toAdd.pop();
		next->CopyTo(fdSet.add_file());
		for (int i = 0; i < next->dependency_count(); ++i)
		{
			const auto &dep = next->dependency(i);
			if (seenDependencies.find(dep->name()) == seenDependencies.end())
			{
				seenDependencies.insert(dep->name());
				toAdd.push(dep);
			}
		}
	}
	return fdSet.SerializeAsString();
}

// 将机器人的装甲板转换为场景实体消息
void convertArmorToSceneEntity(foxglove::SceneEntity* entity, const Robot& robot) {
    for (int i = 0; i < robot.getArmor().size(); ++i) {
        auto pose = robot.getCubePose(i);
        const Eigen::Quaterniond& pose_quaternion = pose.first;
        const Eigen::Vector3d& pose_translation = pose.second;

        auto *cube_msg = entity->add_cubes(); // 添加立方体

        // 设置立方体大小
        auto *size = cube_msg->mutable_size();
        size->set_x(robot.getArmor()[i].size.x());
        size->set_y(robot.getArmor()[i].size.y());
        size->set_z(robot.getArmor()[i].size.z());

        // 设置立方体位置
        auto *position = cube_msg->mutable_pose()->mutable_position();
        position->set_x(pose_translation.x());
        position->set_y(pose_translation.y());
        position->set_z(pose_translation.z());

        // 设置立方体方向
        auto *orientation = cube_msg->mutable_pose()->mutable_orientation();
        orientation->set_w(pose_quaternion.w());
        orientation->set_x(pose_quaternion.x());
        orientation->set_y(pose_quaternion.y());
        orientation->set_z(pose_quaternion.z());

        // 设置立方体颜色
        auto *color = cube_msg->mutable_color();
        color->set_r(robot.getArmor()[i].color.x());
        color->set_g(robot.getArmor()[i].color.y());
        color->set_b(robot.getArmor()[i].color.z());
        color->set_a(robot.getArmor()[i].color.w());
    }
}

void convertCameraToSceneEntity(foxglove::SceneEntity* entity, const Camera& camera) {
	auto pose = camera.getCameraPose();
	const Eigen::Quaterniond& pose_quaternion = pose.first;
	const Eigen::Vector3d& pose_translation = pose.second;

	auto *cube_msg = entity->add_cubes(); // 添加立方体

	// 设置立方体大小
	auto *size = cube_msg->mutable_size();
	size->set_x(camera.getPhysicalSize().x());
	size->set_y(camera.getPhysicalSize().y());
	size->set_z(camera.getPhysicalSize().z());

	// 设置立方体位置
	auto *position = cube_msg->mutable_pose()->mutable_position();
	position->set_x(pose_translation.x());
	position->set_y(pose_translation.y());
	position->set_z(pose_translation.z());

	// 设置立方体方向
	auto *orientation = cube_msg->mutable_pose()->mutable_orientation();
	orientation->set_w(pose_quaternion.w());
	orientation->set_x(pose_quaternion.x());
	orientation->set_y(pose_quaternion.y());
	orientation->set_z(pose_quaternion.z());

	// 设置立方体颜色
	auto *color = cube_msg->mutable_color();
	color->set_r(0);
	color->set_g(1);
	color->set_b(0);
	color->set_a(1);
}

foxglove::FrameTransform createFrameTransformMessage(const std::string& parent_frame_id, const std::string& child_frame_id, 
													 const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation) {
    foxglove::FrameTransform frame_msg;
    frame_msg.set_parent_frame_id(parent_frame_id);
    frame_msg.set_child_frame_id(child_frame_id);
    
    auto* rotation_msg = frame_msg.mutable_rotation();
    rotation_msg->set_x(rotation.x());
    rotation_msg->set_y(rotation.y());
    rotation_msg->set_z(rotation.z());
    rotation_msg->set_w(rotation.w());
    
    auto* translation_msg = frame_msg.mutable_translation();
    translation_msg->set_x(translation.x());
    translation_msg->set_y(translation.y());
    translation_msg->set_z(translation.z());
    
    return frame_msg;
}

bool isYawInRange(const Eigen::Quaterniond& pose_quaternion) {
    // 将四元数转换为旋转矩阵
    Eigen::Matrix3d rotation_matrix = pose_quaternion.toRotationMatrix();
    
    // 提取旋转矩阵中的偏航角（绕 z 轴的旋转）
    double yaw = atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
	// std::cout << yaw << std::endl;
    
    // 检查偏航角是否在给定范围内
    return ((yaw >= M_PI_2 && yaw <= M_PI) || (yaw >= -M_PI && yaw <= -M_PI_2));
}

int main()
{
	// 定义日志处理函数
	const auto logHandler = [](foxglove::WebSocketLogLevel, char const *msg)
	{
		std::cout << msg << std::endl;
	};

	// 创建 WebSocket 服务器实例
	foxglove::ServerOptions serverOptions;
	auto server = foxglove::ServerFactory::createServer<websocketpp::connection_hdl>(
		"C++ Protobuf example server", logHandler, serverOptions);

	// 定义服务器处理程序
	foxglove::ServerHandlers<foxglove::ConnHandle> hdlrs;
	// 订阅处理程序，当有客户端订阅频道时被调用
	hdlrs.subscribeHandler = [&](foxglove::ChannelId chanId, foxglove::ConnHandle)
	{
		std::cout << "first client subscribed to " << chanId << std::endl;
	};
	// 取消订阅处理程序，当最后一个客户端取消订阅频道时被调用
	hdlrs.unsubscribeHandler = [&](foxglove::ChannelId chanId, foxglove::ConnHandle)
	{
		std::cout << "last client unsubscribed from " << chanId << std::endl;
	};
	server->setHandlers(std::move(hdlrs));
	server->start("0.0.0.0", 8765); // 启动服务器，监听 8765 端口

	// 添加频道并序列化频道模式
	const auto channelIds = server->addChannels({
		{
			.topic = "scene_msg",
			.encoding = "protobuf",
			.schemaName = foxglove::SceneUpdate::descriptor()->full_name(),
			.schema = foxglove::base64Encode(SerializeFdSet(foxglove::SceneUpdate::descriptor())),
		},
		{
			.topic = "frame_msg",
			.encoding = "protobuf",
			.schemaName = foxglove::FrameTransforms::descriptor()->full_name(),
			.schema = foxglove::base64Encode(SerializeFdSet(foxglove::FrameTransforms::descriptor())),
		},
		{
			.topic = "img_msg",
			.encoding = "protobuf",
			.schemaName = foxglove::CompressedImage::descriptor()->full_name(),
			.schema = foxglove::base64Encode(SerializeFdSet(foxglove::CompressedImage::descriptor())),
		},
		{
			.topic = "cali_msg",
			.encoding = "protobuf",
			.schemaName = foxglove::CameraCalibration::descriptor()->full_name(),
			.schema = foxglove::base64Encode(SerializeFdSet(foxglove::CameraCalibration::descriptor())),
		}
	});

	// 注册信号处理函数，用于捕获 Ctrl+C 信号
	std::signal(SIGINT, [](int sig)
				{
    std::cerr << "received signal " << sig << ", shutting down" << std::endl;
    running = false; });

	// 初始化敌方机器人
	Eigen::Vector3d initial_position(4.0, 0.0, 0.1); // 初始位置
	double robot_radius = 0.52;						 // 机器人半径
	std::string frame_id = "root";
	Robot enemy_robot(initial_position, robot_radius, frame_id);

	// 初始化自身相机
    Eigen::Matrix3d intrinsic_matrix;
    intrinsic_matrix << 915.120479, 0, 640,
                        0, 915.120479, 512,
                        0, 0, 1;
	Eigen::Matrix3d combined_rotation = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()));
    Eigen::Vector3d translation(0.2, 0, 0.5);
    Camera camera(intrinsic_matrix, combined_rotation, translation);
	camera.setPhysicalSize(Eigen::Vector3d(0.04, 0.04, 0.08));
	camera.setDistortionParams({0, 0, 0, 0});
	camera.printCameraParams();

	// 主循环，持续发送场景更新消息直到接收到关闭信号
	while (running)
	{
		const auto now = nanosecondsSinceEpoch(); // 获取当前时间

		double angle = double(now) / 1e9 * 0.5;
		Eigen::AngleAxisd rotation_vector(angle, Eigen::Vector3d(0, 0, 1));
		Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
		enemy_robot.setOrientation(q);
		foxglove::SceneUpdate scene_msg;				  // 创建场景更新消息
		auto* entity = scene_msg.add_entities();
		*entity->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now); // 设置时间戳
		entity->set_frame_id("root");           // 设置帧 ID
		convertArmorToSceneEntity(entity, enemy_robot);
		convertCameraToSceneEntity(entity, camera);

		foxglove::FrameTransform frame_robot = createFrameTransformMessage("root", "robot", enemy_robot.getOrientation(), enemy_robot.getPosition());
		foxglove::FrameTransform frame_camera = createFrameTransformMessage("root", "camera", camera.getCameraPose().first, camera.getCameraPose().second);
		foxglove::FrameTransform frame_armor = createFrameTransformMessage("root", "armor", enemy_robot.getCubePose(1).first, enemy_robot.getCubePose(1).second);
		foxglove::FrameTransforms frame_msg;
		*frame_msg.add_transforms() = frame_robot;
		*frame_msg.add_transforms() = frame_camera;
		*frame_msg.add_transforms() = frame_armor;


		// 创建一个纯黑色图像
		cv::Mat blackImage = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
		for (int i = 0; i < 4; ++i) {
			auto world_points = enemy_robot.getCube4Point3d(i);
			auto pixel_points = camera.worldToPixel(world_points);
			
			// 不同颜色绘制当前装甲板的像素坐标点
			for (const auto& pixel_point : pixel_points) {
				if (pixel_point.x() >= 0 && pixel_point.x() < IMAGE_WIDTH &&
					pixel_point.y() >= 0 && pixel_point.y() < IMAGE_HEIGHT) {
					cv::Scalar color;
					switch (i) {
						case 0:
							color = cv::Scalar(255, 0, 0); // Blue
							break;
						case 1:
							color = cv::Scalar(0, 255, 0); // Green
							break;
						case 2:
							color = cv::Scalar(0, 0, 255); // Red
							break;
						case 3:
							color = cv::Scalar(255, 255, 0); // Yellow
							break;
						default:
							color = cv::Scalar(255, 255, 255); // White (fallback)
							break;
					}
					if(isYawInRange(enemy_robot.getCubePose(i).first))
						color = cv::Scalar(255, 255, 255);
					cv::circle(blackImage, cv::Point(pixel_point.x(), pixel_point.y()), 4, color, -1);
				}
			}
		}
		// 将图像转换为字节流
		foxglove::CompressedImage img_msg = createCompressedImageMessage(blackImage, "jpeg", "camera");

		foxglove::CameraCalibration cali_msg = createCameraCalibration(camera, IMAGE_WIDTH, IMAGE_HEIGHT, "plumb_bob", "camera", now);
		

		// 广播消息给所有订阅了相应频道的客户端
		// for (const auto& channelId : channelIds){
		// 	server->broadcastMessage(channelId, now, reinterpret_cast<const uint8_t *>(serializedMsg.data()),
		// 							serializedMsg.size());
		// }
		const auto serializedMsg_scene = scene_msg.SerializeAsString(); // 序列化消息
		const auto serializedMsg_frame = frame_msg.SerializeAsString();
		const auto serializedMsg_image = img_msg.SerializeAsString();
		const auto serializedMsg_cali  = cali_msg.SerializeAsString();
		server->broadcastMessage(channelIds[0], now, reinterpret_cast<const uint8_t *>(serializedMsg_scene.data()),
								serializedMsg_scene.size());
		server->broadcastMessage(channelIds[1], now, reinterpret_cast<const uint8_t *>(serializedMsg_frame.data()),
								serializedMsg_frame.size());
		server->broadcastMessage(channelIds[2], now, reinterpret_cast<const uint8_t *>(serializedMsg_image.data()),
								serializedMsg_image.size());
		server->broadcastMessage(channelIds[3], now, reinterpret_cast<const uint8_t *>(serializedMsg_cali.data()),
								serializedMsg_cali.size());
		
		std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 等待 50 毫秒
	}

	// 移除频道并停止服务器
	server->removeChannels(channelIds);
	server->stop();

	return 0;
}
