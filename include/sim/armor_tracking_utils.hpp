#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace sim
{

inline double yawFromRotation(const Eigen::Matrix3d & R)
{
  return std::atan2(R(1, 0), R(0, 0));
}

// Handle armor jump similar to rm_auto_aim Tracker::handleArmorJump:
// reset yaw to measured yaw and recompute center from measured armor pose and radius.
inline void handleArmorJump(
  const Eigen::Vector3d & meas_world, double meas_world_yaw, Eigen::VectorXd & state)
{
  double r = state(8);
  state(6) = meas_world_yaw;  // yaw
  state(0) = meas_world.x() + r * std::cos(meas_world_yaw);  // xc
  state(1) = 0.0;  // vxc
  state(2) = meas_world.y() + r * std::sin(meas_world_yaw);  // yc
  state(3) = 0.0;  // vyc
  state(4) = meas_world.z();  // za
  state(5) = 0.0;  // vza
}

}  // namespace sim
