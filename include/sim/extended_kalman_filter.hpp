#pragma once

#include <Eigen/Dense>

#include <functional>

namespace sim
{

class ExtendedKalmanFilter
{
public:
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  ExtendedKalmanFilter() = default;

  explicit ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h, const VecMatFunc & j_f, const VecMatFunc & j_h,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const Eigen::MatrixXd & P0);

  void setState(const Eigen::VectorXd & x0);
  Eigen::MatrixXd predict();
  Eigen::MatrixXd update(const Eigen::VectorXd & z);

private:
  VecVecFunc f_;
  VecVecFunc h_;
  VecMatFunc jacobian_f_;
  Eigen::MatrixXd F_;
  VecMatFunc jacobian_h_;
  Eigen::MatrixXd H_;
  VoidMatFunc update_Q_;
  Eigen::MatrixXd Q_;
  VecMatFunc update_R_;
  Eigen::MatrixXd R_;
  Eigen::MatrixXd P_pri_;
  Eigen::MatrixXd P_post_;
  Eigen::MatrixXd K_;
  int n_ = 0;
  Eigen::MatrixXd I_;
  Eigen::VectorXd x_pri_;
  Eigen::VectorXd x_post_;
};

}  // namespace sim
