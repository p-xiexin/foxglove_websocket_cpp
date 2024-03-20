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
#include "foxglove/FrameTransform.pb.h"
#include "foxglove/CompressedImage.pb.h"
#include "robot/robot_armor.hpp"

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
void convertArmorToSceneEntity(foxglove::SceneUpdate& msg, const Robot& robot,
                               uint16_t now, const std::string& frame_id) {
    auto *entity = msg.add_entities();
	*entity->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now); // 设置时间戳
    entity->set_frame_id(frame_id);           // 设置帧 ID

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
			.schemaName = foxglove::FrameTransform::descriptor()->full_name(),
			.schema = foxglove::base64Encode(SerializeFdSet(foxglove::FrameTransform::descriptor())),
		},
		{
			.topic = "img_msg",
			.encoding = "protobuf",
			.schemaName = foxglove::CompressedImage::descriptor()->full_name(),
			.schema = foxglove::base64Encode(SerializeFdSet(foxglove::CompressedImage::descriptor())),
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

	// 主循环，持续发送场景更新消息直到接收到关闭信号
	while (running)
	{
		const auto now = nanosecondsSinceEpoch(); // 获取当前时间

		foxglove::SceneUpdate scene_msg;				  // 创建场景更新消息
		double angle = double(now) / 1e9 * 0.5;
		Eigen::AngleAxisd rotation_vector(angle, Eigen::Vector3d(0, 0, 1));
		Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
		enemy_robot.setOrientation(q);
		convertArmorToSceneEntity(scene_msg, enemy_robot, now, frame_id);

		foxglove::FrameTransform frame_msg = createFrameTransformMessage("root", "robot", enemy_robot.getOrientation(), enemy_robot.getPosition());

		// 创建一个纯黑色图像
		int width = 640;
		int height = 480;
		cv::Mat blackImage = cv::Mat::zeros(height, width, CV_8UC3);
		cv::putText(blackImage, "RobotWarrior", cv::Point(width/4, height/2),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		// 将图像转换为字节流
		foxglove::CompressedImage img_msg = createCompressedImageMessage(blackImage, "jpeg", "camera_image");

		// 广播消息给所有订阅了相应频道的客户端
		// for (const auto& channelId : channelIds){
		// 	server->broadcastMessage(channelId, now, reinterpret_cast<const uint8_t *>(serializedMsg.data()),
		// 							serializedMsg.size());
		// }
		const auto serializedMsg_scene = scene_msg.SerializeAsString(); // 序列化消息
		const auto serializedMsg_frame = frame_msg.SerializeAsString();
		const auto serializedMsg_image = img_msg.SerializeAsString();
		server->broadcastMessage(channelIds[0], now, reinterpret_cast<const uint8_t *>(serializedMsg_scene.data()),
								serializedMsg_scene.size());
		server->broadcastMessage(channelIds[1], now, reinterpret_cast<const uint8_t *>(serializedMsg_frame.data()),
								serializedMsg_frame.size());
		server->broadcastMessage(channelIds[2], now, reinterpret_cast<const uint8_t *>(serializedMsg_image.data()),
								serializedMsg_image.size());

		std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 等待 50 毫秒
	}

	// 移除频道并停止服务器
	server->removeChannels(channelIds);
	server->stop();

	return 0;
}

