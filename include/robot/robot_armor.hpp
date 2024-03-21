#include <Eigen/Dense>
#include <vector>

#define CUBE_LENGTH 0.135
#define CUBE_WIDTH 0.124
#define CUBE_HEIGHT 0.019

struct Cube {
    Eigen::Vector4d color = Eigen::Vector4d(0.2, 0.6, 1.0, 1.0); // 颜色，使用Eigen的Vector4d表示，分别表示rgba(归一化)
    Eigen::Vector3d size = Eigen::Vector3d(CUBE_WIDTH, CUBE_LENGTH, CUBE_HEIGHT); // 尺寸，使用Eigen的Vector3d表示
    Eigen::Matrix3d R_bc = Eigen::Matrix3d::Identity(); // 从机器人身体坐标系到装甲板坐标系的变换
    Eigen::Vector3d t_bc = Eigen::Vector3d::Zero(); 
    std::string frame_id;

    Cube(const Eigen::Vector3d& offset, const Eigen::Quaterniond& orientation, const std::string& id)
    : R_bc(orientation.toRotationMatrix()), t_bc(offset), frame_id(id) {}
    Cube(const std::string& id) : frame_id(id) {} // 构造函数初始化frame id
    Cube() {};
};

// euler2Rotation:   body frame to interitail frame
Eigen::Matrix3d euler2Rotation( Eigen::Vector3d  eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);
    double yaw = eulerAngles(2);

    double cr = cos(roll); double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);
    double cy = cos(yaw); double sy = sin(yaw);

    Eigen::Matrix3d RIb;
    RIb<< cy*cp ,   cy*sp*sr - sy*cr,   sy*sr + cy* cr*sp,
            sy*cp,    cy *cr + sy*sr*sp,  sp*sy*cr - cy*sr,
            -sp,         cp*sr,           cp*cr;
    return RIb;
}

class Robot {
public:
    // 构造函数，初始化四个装甲板
    Robot(const Eigen::Vector3d& position, const double& radius, std::string& id)
        : t_wb(position), radius_(radius), frame_id_(id)

    {
        initializeArmor();    
    }

    Robot() {
        initializeArmor();
    };

    // 获取机器人位置
    Eigen::Vector3d getPosition() const {
        return t_wb;
    }

    // 获取机器人速度
    Eigen::Vector3d getVelocity() const {
        return velocity_;
    }

    // 获取机器人姿态
    Eigen::Quaterniond getOrientation() const {
        return Eigen::Quaterniond(R_wb);
    }

    // 获取机器人三轴角速度
    Eigen::Vector3d getAngularVelocity() const {
        return angularVelocity_;
    }

    // 获取装甲板环绕半径
    double getRadius() const {
        return radius_;
    }

    // 获取装甲板信息
    const std::vector<Cube>& getArmor() const {
        return armor_;
    }

    // 获取装甲板位姿
    std::pair<Eigen::Quaterniond, Eigen::Vector3d> getCubePose(int index) const {
        if (index < 0 || index >= armor_.size()) {
            throw std::out_of_range("Invalid armor index");
        }

        const Cube& cube = armor_[index];

        Eigen::Vector3d pose_translation = R_wb * cube.t_bc + t_wb;
        Eigen::Matrix3d pose_rotation = R_wb * cube.R_bc;

        Eigen::Quaterniond pose_quaternion(pose_rotation);
        return std::make_pair(pose_quaternion, pose_translation);
    }

    std::vector<Eigen::Vector3d> getCube4Point3d(int index) const {
        if (index < 0 || index >= armor_.size()) {
            throw std::out_of_range("Invalid armor index");
        }
        const Cube& cube = armor_[index];

        Eigen::Vector3d center = R_wb * cube.t_bc + t_wb; // 计算装甲板中心在世界坐标系中的位置

        // 计算装甲板的上、下、左、右四个点的相对位置
        Eigen::Vector3d left_top_offset(-cube.size.x() / 2, cube.size.y() / 2, 0.0);
        Eigen::Vector3d left_bottom_offset(-cube.size.x() / 2, -cube.size.y() / 2, 0.0);
        Eigen::Vector3d right_bottom_offset(cube.size.x() / 2, -cube.size.y() / 2, 0.0);
        Eigen::Vector3d right_top_offset(cube.size.x() / 2, cube.size.y() / 2, 0.0);

        // 将相对位置转换到世界坐标系中
        Eigen::Vector3d left_top_point = R_wb * cube.R_bc * left_top_offset + center;
        Eigen::Vector3d left_bottom_point = R_wb * cube.R_bc * left_bottom_offset + center;
        Eigen::Vector3d right_bottom_point = R_wb * cube.R_bc * right_bottom_offset + center;
        Eigen::Vector3d right_top_point = R_wb * cube.R_bc * right_top_offset + center;

        // 返回四个点的坐标
        return {left_top_point, left_bottom_point, right_bottom_point, right_top_point};
    }

    // 设置机器人位置
    void setPosition(const Eigen::Vector3d& position) {
        t_wb = position;
    }

    // 设置机器人姿态
    void setOrientation(const Eigen::Quaterniond& orientation) {
        R_wb = orientation.toRotationMatrix();
    }
    
    // 设置机器人移动速度
    void setVelocity(const Eigen::Vector3d& velocity) {
        velocity_ = velocity;
    }

    // 设置机器人旋转速度
    void setAngularVelocity(const Eigen::Vector3d& angularVelocity) {
        angularVelocity_ = angularVelocity;
    }

    // 根据时间间隔t更新机器人位置
    void updatePosition(double dt) {
        // 更新机器人姿态
        Eigen::Matrix3d R_wb_delta = (Eigen::AngleAxisd(angularVelocity_.x() * dt, Eigen::Vector3d::UnitX()) *
                                       Eigen::AngleAxisd(angularVelocity_.y() * dt, Eigen::Vector3d::UnitY()) *
                                       Eigen::AngleAxisd(angularVelocity_.z() * dt, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        R_wb = R_wb * R_wb_delta;

        // 更新机器人位置
        t_wb += velocity_ * dt;
    }

private:
    std::vector<Cube> armor_;

    double radius_ = 0.21; // 机器人装甲板环绕半径
    Eigen::Matrix3d R_wb = Eigen::Matrix3d::Identity(); // 从世界坐标系到机器人身体坐标系的变换
    Eigen::Vector3d t_wb = Eigen::Vector3d::Zero(); 

    Eigen::Vector3d velocity_ = Eigen::Vector3d::Zero(); // 机器人速度
    Eigen::Vector3d angularVelocity_ = Eigen::Vector3d::Zero(); // 机器人三轴角速度

    std::string frame_id_;

    // 初始化装甲板
    void initializeArmor() {
        double radius = radius_;
        // 计算装甲板相对于机体的位置
        Eigen::Vector3d front_offset(radius, 0.0, 0.0);
        Eigen::Vector3d left_offset(0.0, radius, 0.0);
        Eigen::Vector3d back_offset(-radius, 0.0, 0.0);
        Eigen::Vector3d right_offset(0.0, -radius, 0.0);
        
        // 计算装甲板相对于机体的姿态
        double armor_pitch = M_PI/2 -M_PI/20;
        Eigen::Quaterniond front_orientation(euler2Rotation(Eigen::Vector3d(0.0, armor_pitch, 0.0)));
        Eigen::Quaterniond left_orientation(euler2Rotation(Eigen::Vector3d(0.0, armor_pitch, M_PI_2)));
        Eigen::Quaterniond back_orientation(euler2Rotation(Eigen::Vector3d(0.0, armor_pitch, M_PI)));
        Eigen::Quaterniond right_orientation(euler2Rotation(Eigen::Vector3d(0.0, armor_pitch, -M_PI_2)));
        // Eigen::Quaterniond front_orientation(Eigen::AngleAxisd(armor_pitch, Eigen::Vector3d::UnitY()));

        // Eigen::Quaterniond left_orientation(Eigen::AngleAxisd(armor_pitch, Eigen::Vector3d::UnitY()) *
        //                                     Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));

        // Eigen::Quaterniond back_orientation(Eigen::AngleAxisd(armor_pitch, Eigen::Vector3d::UnitY()) *
        //                                     Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));

        // Eigen::Quaterniond right_orientation(Eigen::AngleAxisd(armor_pitch, Eigen::Vector3d::UnitY()) *
        //                                     Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()));

        armor_.emplace_back(front_offset, front_orientation, frame_id_);
        armor_.emplace_back(left_offset, left_orientation, frame_id_);
        armor_.emplace_back(back_offset, back_orientation, frame_id_);
        armor_.emplace_back(right_offset, right_orientation, frame_id_);
    }
};
