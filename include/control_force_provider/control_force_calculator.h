#pragma once
#define BOOST_THREAD_PROVIDES_FUTURE

#include <ros/ros.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>

#include "control_force_provider/python_context.h"
#include "obstacle.h"

namespace control_force_provider::backend {
class ControlForceCalculator {
 private:
  bool goal_available_ = false;

 protected:
  Eigen::Vector4d ee_position;
  Eigen::Vector3d rcm;
  Eigen::Vector4d goal;
  boost::shared_ptr<Obstacle> obstacle;
  friend class Visualizer;

  virtual void getForceImpl(Eigen::Vector4d& force) = 0;

 public:
  explicit ControlForceCalculator(boost::shared_ptr<Obstacle>& obstacle) : obstacle(obstacle) {}
  void getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position_) {
    if (!goal_available_) setGoal(ee_position);
    ee_position = ee_position_;
    getForceImpl(force);
  }
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm; }
  void setRCM(const Eigen::Vector3d& rcm_) { rcm = rcm_; };
  [[nodiscard]] const Eigen::Vector4d& getGoal() const { return goal; }
  void setGoal(const Eigen::Vector4d& goal_) {
    goal_available_ = true;
    goal = goal_;
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

class ReinforcementLearningAgent : public ControlForceCalculator, PythonEnvironment {
 private:
  const ros::Duration interval_duration_;
  Eigen::Vector4d current_force_;
  ros::Time last_calculation_;
  boost::future<Eigen::Vector4d> calculation_future_;
  boost::promise<Eigen::Vector4d> calculation_promise_;

  void calculationRunnable();

 protected:
  PythonObject networks_module;
  virtual Eigen::Vector4d getAction() = 0;

 public:
  ReinforcementLearningAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
  void getForceImpl(Eigen::Vector4d& force) override;
  ~ReinforcementLearningAgent() override = default;
};

class DeepQNetworkAgent : public ReinforcementLearningAgent {
 protected:
  Eigen::Vector4d getAction() override;

 public:
  DeepQNetworkAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
};
}  // namespace control_force_provider::backend