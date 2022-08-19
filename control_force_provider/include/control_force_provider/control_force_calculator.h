#pragma once
#define BOOST_THREAD_PROVIDES_FUTURE

#include <ros/ros.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <deque>

#include "control_force_provider_msgs/UpdateNetwork.h"
#include "obstacle.h"

namespace control_force_provider::backend {
class ControlForceCalculator {
 private:
  bool rcm_available_ = false;
  bool goal_available_ = false;

 protected:
  Eigen::Vector4d ee_position;
  Eigen::Vector3d rcm;
  Eigen::Vector4d goal;
  boost::shared_ptr<Obstacle> obstacle;
  Eigen::Vector4d ob_position;
  Eigen::Vector3d ob_rcm;
  friend class Visualizer;

  virtual void getForceImpl(Eigen::Vector4d& force) = 0;

 public:
  explicit ControlForceCalculator(boost::shared_ptr<Obstacle>& obstacle) : obstacle(obstacle) {}
  void getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position_) {
    if (!goal_available_) setGoal(ee_position);
    if (rcm_available_) {
      ee_position = ee_position_;
      obstacle->getPosition(ob_position);
      ob_rcm = obstacle->getRCM();
      getForceImpl(force);
    }
  }
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm; }
  void setRCM(const Eigen::Vector3d& rcm_) {
    rcm_available_ = true;
    rcm = rcm_;
  };
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

 protected:
  void getForceImpl(Eigen::Vector4d& force) override;

 public:
  PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
  ~PotentialFieldMethod() override = default;
};

class StateProvider {
 private:
  int state_dim_ = 0;
  static inline const std::string pattern_regex = "[[:alpha:]]{3}\\(([[:alpha:]][[:digit:]]+)*\\)";
  static inline const std::string arg_regex = "[[:alpha:]][[:digit:]]+";
  class StatePopulator {
   private:
    const double* vector_;
    const int length_;
    std::deque<Eigen::VectorXd> history_;
    unsigned int stride_index_ = 0;

   public:
    unsigned int history_length_ = 1;
    unsigned int history_stride_ = 0;
    StatePopulator(const double* vector, int length = 0) : vector_(vector), length_(length){};
    void populate(torch::Tensor& state, int& index);
    int getDim() const { return length_ * history_length_; };
  };
  std::vector<StatePopulator> state_populators_;

 public:
  StateProvider(const Eigen::Vector4d& ee_position, const Eigen::Vector3d& robot_rcm, const Eigen::Vector4d& obstacle_position,
                const Eigen::Vector3d& obstacle_rcm, const Eigen::Vector4d& goal, const std::string& state_pattern);
  torch::Tensor createState();
  ~StateProvider() = default;
  int getStateDim() const { return state_dim_; };
};

class ReinforcementLearningAgent : public ControlForceCalculator {
 private:
  const bool train;
  const ros::Duration interval_duration_;
  Eigen::Vector4d current_force_;
  ros::Time last_calculation_;
  boost::future<Eigen::Vector4d> calculation_future_;
  boost::promise<Eigen::Vector4d> calculation_promise_;

  Eigen::Vector4d getAction();
  void calculationRunnable();

 protected:
  const std::string output_dir;
  boost::shared_ptr<StateProvider> state_provider;
  boost::shared_ptr<ros::ServiceClient> training_service_client;

  virtual torch::Tensor getActionInference(torch::Tensor& state) = 0;
  void getForceImpl(Eigen::Vector4d& force) override;

 public:
  ReinforcementLearningAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config, ros::NodeHandle& node_handle);
  ~ReinforcementLearningAgent() override = default;
};

class DeepQNetworkAgent : public ReinforcementLearningAgent {
 protected:
  torch::Tensor getActionInference(torch::Tensor& state) override;

 public:
  DeepQNetworkAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config, ros::NodeHandle& node_handle);
};
}  // namespace control_force_provider::backend