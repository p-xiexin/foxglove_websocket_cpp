#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <net/if.h>

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
static uint64_t nanosecondsSinceEpoch();
std::string getLocalIPAddress();
// 将指定描述符及其所有依赖项序列化为字符串，用作通道模式
static std::string SerializeFdSet(const google::protobuf::Descriptor *toplevelDescriptor);
// 将机器人的装甲板转换为场景实体消息
static void convertArmorToSceneEntity(foxglove::SceneEntity* entity, const Robot& robot);
// 将相机转换为场景实体消息
static void convertCameraToSceneEntity(foxglove::SceneEntity* entity, const Camera& camera);
// 创建压缩图像消息
static foxglove::CompressedImage createCompressedImageMessage(const cv::Mat& image, const std::string& format, const std::string& frame_id);
// 创建相机标定消息
static foxglove::CameraCalibration createCameraCalibration(const Camera& camera, int image_width, int image_height, 
													const std::string& distortion_model, const std::string& frame_id);
// 创建坐标变换消息
static foxglove::FrameTransform createFrameTransformMessage(const std::string& parent_frame_id, const std::string& child_frame_id, 
													 const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation);
// 检查偏航角是否在指定范围内
static bool isYawInRange(const Eigen::Quaterniond& pose_quaternion);
// 绘制装甲板的像素坐标点
static void drawArmorPoints(const Robot& enemy_robot, const Camera& camera, cv::Mat& image);

// 返回机器人看到的装甲板四个角点
std::vector<std::vector<Eigen::Vector2d>> get_armorPixelPoints();


const double robot_radius = 0.52;						 // 机器人半径
const Eigen::Vector3d initial_position(4.0, 0.0, 0.1); // 机器人初始位置
// 相机内外参
const Eigen::Matrix3d intrinsic_matrix = [] {
    Eigen::Matrix3d matrix;
    matrix << 915.120479, 0, 640,
              0, 915.120479, 512,
              0, 0, 1;
    return matrix;
}();
const Eigen::Matrix3d rotation = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()));
const Eigen::Vector3d translation(0.2, 0, 0.5);

std::vector<std::vector<Eigen::Vector2d>> armorPixelPoints;

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
	std::string ipAddress = "0.0.0.0";
    try {
        ipAddress = getLocalIPAddress();
        std::cout << "Local IP Address: " << ipAddress << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        // return 1;
    }
	server->setHandlers(std::move(hdlrs));
	server->start(ipAddress, 8765); // 启动服务器，监听 8765 端口

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
	Robot enemy_robot(initial_position, robot_radius, "root");

	// 初始化自身相机
    Camera camera(intrinsic_matrix, rotation, translation);
	camera.setPhysicalSize(Eigen::Vector3d(0.04, 0.04, 0.08));
	camera.setDistortionParams({0, 0, 0, 0});

	// 主循环，持续发送场景更新消息直到接收到关闭信号
	while (running)
	{
		std::vector<std::string> serializedMsgs;
		const auto now = nanosecondsSinceEpoch(); // 获取当前时间

		// 更新机器人状态
		enemy_robot.setAngularVelocity(Eigen::Vector3d(0, 0, M_PI_2));
		enemy_robot.updatePosition(0.05);

		// channel0：场景信息
		foxglove::SceneUpdate scene_msg;
		auto* entity = scene_msg.add_entities();
		*entity->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now); // 设置时间戳
		entity->set_frame_id("root");           // 设置帧 ID
		convertArmorToSceneEntity(entity, enemy_robot);
		convertCameraToSceneEntity(entity, camera);
		serializedMsgs.emplace_back(scene_msg.SerializeAsString());

		// channel1：坐标变换
		foxglove::FrameTransform frame_robot = createFrameTransformMessage("root", "robot", enemy_robot.getOrientation(), enemy_robot.getPosition());
		foxglove::FrameTransform frame_camera = createFrameTransformMessage("root", "camera", camera.getCameraPose().first, camera.getCameraPose().second);
		foxglove::FrameTransform frame_armor = createFrameTransformMessage("root", "armor", enemy_robot.getCubePose(1).first, enemy_robot.getCubePose(1).second);
		foxglove::FrameTransforms frame_msg;
		*frame_msg.add_transforms() = frame_robot;
		*frame_msg.add_transforms() = frame_camera;
		*frame_msg.add_transforms() = frame_armor;
		serializedMsgs.emplace_back(frame_msg.SerializeAsString());

		// channel2：图像
		cv::Mat blackImage = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
		drawArmorPoints(enemy_robot, camera, blackImage);
		foxglove::CompressedImage img_msg = createCompressedImageMessage(blackImage, "jpeg", "camera");
		serializedMsgs.emplace_back(img_msg.SerializeAsString());

		// channel3：相机参数
		foxglove::CameraCalibration cali_msg = createCameraCalibration(camera, IMAGE_WIDTH, IMAGE_HEIGHT, "plumb_bob", "camera");
		serializedMsgs.emplace_back(cali_msg.SerializeAsString());

		// 广播消息给所有订阅了相应频道的客户端
		for (int i = 0; i < channelIds.size(); i++){
			server->broadcastMessage(channelIds[i], now, reinterpret_cast<const uint8_t *>(serializedMsgs[i].data()),
									serializedMsgs[i].size());
		}
		
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}

	// 移除频道并停止服务器
	server->removeChannels(channelIds);
	server->stop();

	return 0;
}

static uint64_t nanosecondsSinceEpoch()
{
	return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
						std::chrono::system_clock::now().time_since_epoch())
						.count());
}

std::string getLocalIPAddress() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        throw std::runtime_error("Error creating socket");
    }

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    ifr.ifr_addr.sa_family = AF_INET;

    // 获取第一个非回环设备的IP地址
    strncpy(ifr.ifr_name, "eth0", IFNAMSIZ-1);

    if (ioctl(sockfd, SIOCGIFADDR, &ifr) < 0) {
        close(sockfd);
        throw std::runtime_error("Error getting IP address");
    }

    close(sockfd);

    struct sockaddr_in* addr = reinterpret_cast<struct sockaddr_in*>(&ifr.ifr_addr);
    return inet_ntoa(addr->sin_addr);
}

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

static void convertArmorToSceneEntity(foxglove::SceneEntity* entity, const Robot& robot) {
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

static void convertCameraToSceneEntity(foxglove::SceneEntity* entity, const Camera& camera) {
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

static foxglove::CameraCalibration createCameraCalibration(const Camera& camera, int image_width, int image_height, const std::string& distortion_model, const std::string& frame_id) {
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
	const auto now = nanosecondsSinceEpoch(); // 获取当前时间
    *calibration.mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now);

    // 设置帧ID
    calibration.set_frame_id(frame_id);

    return calibration;
}


static foxglove::FrameTransform createFrameTransformMessage(const std::string& parent_frame_id, const std::string& child_frame_id, 
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

	const auto now = nanosecondsSinceEpoch(); // 获取当前时间
	*frame_msg.mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now);
    
    return frame_msg;
}

static foxglove::CompressedImage createCompressedImageMessage(const cv::Mat& image, const std::string& format, const std::string& frame_id) {
    // 将图像编码为字节流
    std::vector<uchar> buf;
    cv::imencode(".png", image, buf);
    std::string imageBytes(reinterpret_cast<char*>(buf.data()), buf.size());

    // 创建 CompressedImage 消息并设置相关参数
    foxglove::CompressedImage img_msg;
    img_msg.set_format(format);
    img_msg.set_frame_id(frame_id);
    img_msg.set_data(imageBytes);

	const auto now = nanosecondsSinceEpoch(); // 获取当前时间
	*img_msg.mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now);

    return img_msg;
}

static bool isYawInRange(const Eigen::Quaterniond& pose_quaternion) {
    Eigen::Matrix3d rotation_matrix = pose_quaternion.toRotationMatrix();
    
    // 提取旋转矩阵中的偏航角（绕 z 轴的旋转）
    double yaw = atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
	// std::cout << yaw << std::endl;
    
    // 检查偏航角是否在给定范围内
	return (std::abs(yaw) >= 2*M_PI/3);
    // return ((yaw >= M_PI_2 && yaw <= M_PI) || (yaw >= -M_PI && yaw <= -M_PI_2));
}

static void drawArmorPoints(const Robot& enemy_robot, const Camera& camera, cv::Mat& image) {
	armorPixelPoints.clear();

    for (int i = 0; i < 4; ++i) {
        auto world_points = enemy_robot.getCube4Point3d(i);
        auto pixel_points = camera.worldToPixel(world_points);
        
        // 不同颜色绘制当前装甲板的像素坐标点
		cv::Scalar color;
        for (const auto& pixel_point : pixel_points) {
            if (pixel_point.x() >= 0 && pixel_point.x() < image.cols &&
                pixel_point.y() >= 0 && pixel_point.y() < image.rows) {
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
                cv::circle(image, cv::Point(pixel_point.x(), pixel_point.y()), 4, color, -1);
            }
        }

		if (isYawInRange(enemy_robot.getCubePose(i).first)){
			cv::line(image, cv::Point(pixel_points[0].x(), pixel_points[0].y()), cv::Point(pixel_points[2].x(), pixel_points[2].y()), color, 2);
			cv::line(image, cv::Point(pixel_points[1].x(), pixel_points[0].y()), cv::Point(pixel_points[3].x(), pixel_points[2].y()), color, 2);
			armorPixelPoints.push_back(pixel_points);
		}
    }
}

std::vector<std::vector<Eigen::Vector2d>> get_armorPixelPoints(){
	return armorPixelPoints;
}
