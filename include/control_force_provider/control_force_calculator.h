#pragma once
#define BOOST_THREAD_PROVIDES_FUTURE

#include <ros/ros.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>

#include "obstacle.h"

namespace control_force_provider::backend {
class ControlForceCalculator {
 protected:
  Eigen::Vector4d ee_position_;
  Eigen::Vector3d rcm_;
  Eigen::Vector4d goal_;
  bool goal_available_ = false;
  boost::shared_ptr<Obstacle> obstacle_;
  friend class Visualizer;

  virtual void getForceImpl(Eigen::Vector4d& force) = 0;

 public:
  explicit ControlForceCalculator(boost::shared_ptr<Obstacle>& obstacle) : obstacle_(obstacle) {}
  void getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position) {
    if (!goal_available_) setGoal(ee_position);
    ee_position_ = ee_position;
    getForceImpl(force);
  }
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm_; }
  void setRCM(const Eigen::Vector3d& rcm) { rcm_ = rcm; };
  [[nodiscard]] const Eigen::Vector4d& getGoal() const { return goal_; }
  void setGoal(const Eigen::Vector4d& goal) {
    goal_available_ = true;
    goal_ = goal;
  };
};

class PotentialFieldMethod : public ControlForceCalculator {
 private:
  const double attraction_strength_;
  const double attraction_distance_;
  const double repulsion_strength_;
  const double repulsion_distance_;
  const double z_translation_strength_;
  const double min_rcm_distance_;
  Eigen::Vector3d point_on_l1_;
  Eigen::Vector3d point_on_l2_;
  friend class Visualizer;

 public:
  PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
  void getForceImpl(Eigen::Vector4d& force) override;
  ~PotentialFieldMethod() override = default;
};

class ReinforcementLearningAgent : public ControlForceCalculator {
 private:
  const ros::Duration interval_duration_;
  const bool train_;
  Eigen::Vector4d current_force_ = {0, 0, 0, 0};
  ros::Time last_calculation_;
  boost::future<Eigen::Vector4d> calculation_future_;
  boost::promise<Eigen::Vector4d> calculation_promise_;

  void calculationRunnable();

 protected:
  virtual Eigen::Vector4d getAction();

 public:
  ReinforcementLearningAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
  void getForceImpl(Eigen::Vector4d& force) override;
  ~ReinforcementLearningAgent() override = default;
};
}  // namespace control_force_provider::backend