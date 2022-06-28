#pragma once

#include <ros/ros.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

#include "obstacle.h"

namespace control_force_provider::backend {
class ControlForceCalculator {
 protected:
  Eigen::Vector3d rcm_;
  Eigen::Vector3d goal_;
  bool goal_available_ = false;
  boost::shared_ptr<Obstacle> obstacle_;
  friend class Visualizer;

  virtual void getForceImpl(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position) = 0;

 public:
  explicit ControlForceCalculator(boost::shared_ptr<Obstacle>& obstacle) : obstacle_(obstacle) {}
  void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position) {
    if (goal_available_)
      getForceImpl(force, ee_position);
    else
      force = {};
  }
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm_; }
  void setRCM(const Eigen::Vector3d& rcm) {
    rcm_ = rcm;
    goal_available_ = true;
    setGoal(rcm_);
  };
  [[nodiscard]] const Eigen::Vector3d& getGoal() const { return goal_; }
  void setGoal(const Eigen::Vector3d& goal) {
    ROS_WARN_STREAM("Set goal");
    goal_available_ = true;
    goal_ = goal;
  };
};

class PotentialFieldMethod : public ControlForceCalculator {
 private:
  double attraction_strength_;
  double attraction_distance_;
  double repulsion_strength_;
  double repulsion_distance_;
  Eigen::Vector3d point_on_l1_;
  Eigen::Vector3d point_on_l2_;
  friend class Visualizer;

 public:
  PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
  void getForceImpl(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position) override;
  ~PotentialFieldMethod() override = default;
};
}  // namespace control_force_provider::backend