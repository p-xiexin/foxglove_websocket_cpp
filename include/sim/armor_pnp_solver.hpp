#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace sim
{

// PnP solver for planar armor based on rm_auto_aim logic (IPPE)
class ArmorPnPSolver
{
public:
  ArmorPnPSolver(
    const Eigen::Matrix3d & intrinsic, const std::vector<double> & distortion, double armor_width,
    double armor_height);

  bool solve(const std::vector<Eigen::Vector2d> & image_points, cv::Mat & rvec, cv::Mat & tvec) const;

  static double rvecToYaw(const cv::Mat & rvec);

private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3f> armor_points_;
};

}  // namespace sim
