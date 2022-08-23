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
  std::vector<boost::shared_ptr<Obstacle>> obstacles;
  std::vector<Eigen::Vector4d> ob_positions;
  std::vector<Eigen::Vector3d> ob_rcms;
  friend class Visualizer;
  friend class StateProvider;

  virtual void getForceImpl(Eigen::Vector4d& force) = 0;

 public:
  explicit ControlForceCalculator(std::vector<boost::shared_ptr<Obstacle>> obstacles_);
  void getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position_);
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
  std::vector<Eigen::Vector3d> points_on_l1_;
  std::vector<Eigen::Vector3d> points_on_l2_;
  friend class Visualizer;

 protected:
  void getForceImpl(Eigen::Vector4d& force) override;

 public:
  PotentialFieldMethod(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config);
  ~PotentialFieldMethod() override = default;
};

class StateProvider {
 private:
  unsigned int state_dim_ = 0;
  static inline const std::string pattern_regex = "[[:alpha:]]{3}\\(([[:alpha:]][[:digit:]]+)*\\)";
  static inline const std::string arg_regex = "[[:alpha:]][[:digit:]]+";
  class StatePopulator {
   private:
    std::vector<std::deque<Eigen::VectorXd>> histories_;
    unsigned int stride_index_ = 0;

   public:
    std::vector<const double*> vectors_;
    int length_ = 0;
    unsigned int history_length_ = 1;
    unsigned int history_stride_ = 0;
    StatePopulator() = default;
    ;
    void populate(torch::Tensor& state, int& index);
    [[nodiscard]] unsigned int getDim() const { return length_ * vectors_.size() * history_length_; };
  };
  std::vector<StatePopulator> state_populators_;
  static StatePopulator createPopulatorFromString(const ControlForceCalculator& cfc, const std::string& str);

 public:
  StateProvider(const ControlForceCalculator& cfc, const std::string& state_pattern);
  torch::Tensor createState();
  ~StateProvider() = default;
  [[nodiscard]] unsigned int getStateDim() const { return state_dim_; };
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
  ReinforcementLearningAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle);
  ~ReinforcementLearningAgent() override = default;
};

class DeepQNetworkAgent : public ReinforcementLearningAgent {
 protected:
  torch::Tensor getActionInference(torch::Tensor& state) override;

 public:
  DeepQNetworkAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle);
};
}  // namespace control_force_provider::backend