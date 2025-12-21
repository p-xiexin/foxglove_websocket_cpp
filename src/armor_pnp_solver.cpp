#include "sim/armor_pnp_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>

namespace sim
{

ArmorPnPSolver::ArmorPnPSolver(
  const Eigen::Matrix3d & intrinsic, const std::vector<double> & distortion, double armor_width,
  double armor_height)
: camera_matrix_(cv::Mat(3, 3, CV_64F)),
  dist_coeffs_(cv::Mat::zeros(1, 5, CV_64F))
{
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      camera_matrix_.at<double>(i, j) = intrinsic(i, j);
    }
  }
  for (size_t i = 0; i < std::min<size_t>(5, distortion.size()); ++i) {
    dist_coeffs_.at<double>(i) = distortion[i];
  }

  const double half_w = armor_width * 0.5;
  const double half_h = armor_height * 0.5;
  // Points are on armor plane (z=0), matching getCube4Point3d order: left-top, left-bottom, right-bottom, right-top
  armor_points_.emplace_back(cv::Point3f(-half_w, half_h, 0.0));   // left-top
  armor_points_.emplace_back(cv::Point3f(-half_w, -half_h, 0.0));  // left-bottom
  armor_points_.emplace_back(cv::Point3f(half_w, -half_h, 0.0));   // right-bottom
  armor_points_.emplace_back(cv::Point3f(half_w, half_h, 0.0));    // right-top
}

bool ArmorPnPSolver::solve(const std::vector<Eigen::Vector2d> & image_points, cv::Mat & rvec, cv::Mat & tvec) const
{
  if (image_points.size() != 4) {
    return false;
  }

  std::vector<cv::Point2f> cv_image_points;
  cv_image_points.reserve(4);
  for (const auto & pt : image_points) {
    cv_image_points.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
  }

  if (std::any_of(cv_image_points.begin(), cv_image_points.end(), [](const cv::Point2f & p) {
        return p.x < 0 || p.y < 0;
      })) {
    return false;
  }

  return cv::solvePnP(
    armor_points_, cv_image_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
}

double ArmorPnPSolver::rvecToYaw(const cv::Mat & rvec)
{
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  return std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
}

}  // namespace sim
