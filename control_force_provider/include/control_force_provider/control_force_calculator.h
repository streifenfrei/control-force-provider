#pragma once
#define BOOST_THREAD_PROVIDES_FUTURE

#include <ros/ros.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <deque>

#include "control_force_provider_msgs/UpdateNetwork.h"
#include "obstacle.h"
#include "time.h"
#include "utils.h"

using namespace control_force_provider;

namespace control_force_provider::backend {
class ControlForceCalculator {
 private:
  const inline static double workspace_bb_stopping_strength = 0.001;
  const inline static double rcm_max_norm = 10;
  bool rcm_available_ = false;
  bool goal_available_ = false;

 protected:
  torch::Tensor workspace_bb_origin_;
  const torch::Tensor workspace_bb_dims_;
  const torch::Scalar max_force_;
  torch::Tensor ee_position;
  torch::Tensor ee_rotation;
  torch::Tensor ee_velocity;
  torch::Tensor rcm;
  torch::Tensor goal;
  double start_time;
  torch::Scalar elapsed_time;
  std::vector<boost::shared_ptr<Obstacle>> obstacles;
  boost::shared_ptr<ObstacleLoader> obstacle_loader_;
  std::vector<torch::Tensor> ob_positions;
  std::vector<torch::Tensor> ob_rotations;
  std::vector<torch::Tensor> ob_velocities;
  std::vector<torch::Tensor> ob_rcms;
  std::vector<torch::Tensor> points_on_l1_;
  std::vector<torch::Tensor> points_on_l2_;
  torch::Tensor offset_;
  friend class Visualizer;
  friend class StateProvider;

  virtual void getForceImpl(Eigen::Vector3d& force) = 0;

  void updateDistanceVectors();

  void setOffset(Eigen::Vector3d offset);

 public:
  ControlForceCalculator(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, const std::string& data_path);
  void getForce(Eigen::Vector3d& force, const Eigen::Vector3d& ee_position_);
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] Eigen::Vector3d getRCM() const {
    auto acc = rcm.accessor<double, 1>();
    return Eigen::Vector3d(acc[0], acc[1], acc[2]);
  }
  void setRCM(const Eigen::Vector3d& rcm_) {
    if (rcm_available_) return;  // disable function once rcm is set: workaround to avoid race condition
    if (rcm_.norm() < rcm_max_norm) {
      torch::Tensor rcm_t = utils::vectorToTensor(rcm_);
      rcm_available_ = true;
      rcm = rcm_t - offset_;
    }
  };
  [[nodiscard]] Eigen::Vector3d getGoal() const {
    auto acc = goal.accessor<double, 1>();
    return Eigen::Vector3d(acc[0], acc[1], acc[2]);
  }

  void setGoal(const Eigen::Vector3d& goal_) {
    goal_available_ = true;
    goal = utils::vectorToTensor(goal_);
    goal -= offset_;
    start_time = Time::now();
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
  friend class Visualizer;

 protected:
  void getForceImpl(Eigen::Vector3d& force) override;

 public:
  PotentialFieldMethod(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, const std::string& data_path = "");
  ~PotentialFieldMethod() override = default;
};

class StateProvider {
 private:
  unsigned int state_dim_ = 0;
  static inline const std::string pattern_regex = "[[:alpha:]]{3}\\(([[:alpha:]][[:digit:]]+)*\\)";
  static inline const std::string arg_regex = "[[:alpha:]][[:digit:]]+";
  class StatePopulator {
   private:
    std::vector<std::deque<torch::Tensor>> histories_;
    unsigned int stride_index_ = 0;

   public:
    std::vector<const torch::Tensor*> tensors_;
    int length_ = 0;
    unsigned int history_length_ = 1;
    unsigned int history_stride_ = 0;
    StatePopulator() = default;
    void populate(torch::Tensor& state, int& index);
    [[nodiscard]] unsigned int getDim() const { return length_ * tensors_.size() * history_length_; };
  };
  std::vector<StatePopulator> state_populators_;
  static StatePopulator createPopulatorFromString(const ControlForceCalculator& cfc, const std::string& str);

 public:
  StateProvider(const ControlForceCalculator& cfc, const std::string& state_pattern);
  torch::Tensor createState();
  ~StateProvider() = default;
  [[nodiscard]] unsigned int getStateDim() const { return state_dim_; };
};

class EpisodeContext {
 private:
  Eigen::Vector3d start_;
  Eigen::Vector3d goal_;
  std::vector<boost::shared_ptr<Obstacle>>& obstacles_;
  boost::shared_ptr<ObstacleLoader>& obstacle_loader_;
  boost::random::mt19937 rng_;
  const torch::Tensor start_bb_origin;
  const torch::Tensor start_bb_dims;
  const torch::Tensor goal_bb_origin;
  const torch::Tensor goal_bb_dims;
  double begin_max_offset_;
  friend class Visualizer;

 public:
  EpisodeContext(std::vector<boost::shared_ptr<Obstacle>>& obstacles_, boost::shared_ptr<ObstacleLoader>& obstacle_loader, const YAML::Node& config);
  void generateEpisode();
  void startEpisode();
  const Eigen::Vector3d& getStart() const { return start_; };
  const Eigen::Vector3d& getGoal() const { return goal_; };
};

class ReinforcementLearningAgent : public ControlForceCalculator {
 private:
  const inline static double transition_smoothness = 0.001;
  const inline static unsigned int goal_delay = 10;
  const bool train;
  const double interval_duration_;
  const double goal_reached_threshold_distance_;
  const bool rcm_origin_;
  EpisodeContext episode_context_;
  bool initializing_episode = true;
  Eigen::Vector3d current_force_;
  double last_calculation_;
  boost::future<Eigen::Vector3d> calculation_future_;
  boost::promise<Eigen::Vector3d> calculation_promise_;
  unsigned int goal_delay_count = 0;
  friend class Visualizer;

  Eigen::Vector3d getAction();
  void calculationRunnable();

 protected:
  const std::string output_dir;
  boost::shared_ptr<StateProvider> state_provider;
  boost::shared_ptr<ros::ServiceClient> training_service_client;

  virtual torch::Tensor getActionInference(torch::Tensor& state) = 0;
  void getForceImpl(Eigen::Vector3d& force) override;

 public:
  ReinforcementLearningAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle,
                             const std::string& data_path);
  ~ReinforcementLearningAgent() override = default;
};

class DeepQNetworkAgent : public ReinforcementLearningAgent {
 protected:
  torch::Tensor getActionInference(torch::Tensor& state) override;

 public:
  DeepQNetworkAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle, const std::string& data_path);
};

class MonteCarloAgent : public ReinforcementLearningAgent {
 protected:
  torch::Tensor getActionInference(torch::Tensor& state) override;

 public:
  MonteCarloAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle, const std::string& data_path);
};
}  // namespace control_force_provider::backend