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
class Environment {
 private:
  const inline static double workspace_bb_stopping_strength = 0.001;

  void updateDistanceVectors();

 public:
  torch::Tensor workspace_bb_origin;
  const torch::Tensor workspace_bb_dims;
  const torch::Scalar max_force;
  torch::Tensor ee_position;
  torch::Tensor ee_rotation;
  torch::Tensor ee_velocity;
  torch::Tensor rcm;
  torch::Tensor goal;
  double start_time;
  torch::Scalar elapsed_time;
  std::vector<boost::shared_ptr<Obstacle>> obstacles;
  boost::shared_ptr<ObstacleLoader> obstacle_loader;
  std::vector<torch::Tensor> ob_positions;
  std::vector<torch::Tensor> ob_rotations;
  std::vector<torch::Tensor> ob_velocities;
  std::vector<torch::Tensor> ob_rcms;
  std::vector<torch::Tensor> points_on_l1_;
  std::vector<torch::Tensor> points_on_l2_;
  torch::Tensor offset;
  Environment(const YAML::Node& config);
  void setOffset(const torch::Tensor& offset_);
  void update(const torch::Tensor& ee_position_);
  void clipForce(torch::Tensor& force_);
};

class ControlForceCalculator {
 private:
  const inline static double rcm_max_norm = 10;
  bool rcm_available_ = false;
  bool goal_available_ = false;

 protected:
  Environment env;
  friend class Visualizer;
  friend class StateProvider;

  virtual void getForceImpl(torch::Tensor& force) = 0;

 public:
  ControlForceCalculator(const YAML::Node& config);
  void getForce(torch::Tensor& force, const torch::Tensor& ee_position_);
  [[nodiscard]] torch::Tensor getRCM() const { return env.rcm; }
  void setRCM(const torch::Tensor& rcm_) {
    if (utils::norm(rcm_).item().toDouble() < rcm_max_norm) {
      torch::Tensor rcm_t = rcm_;
      rcm_available_ = true;
      env.rcm = rcm_t - env.offset;
    }
  };
  [[nodiscard]] torch::Tensor getGoal() const { return env.goal; }

  void setGoal(const torch::Tensor& goal_) {
    goal_available_ = true;
    env.goal = goal_ - env.offset;
    env.start_time = Time::now();
  };
  virtual ~ControlForceCalculator() = default;

  static boost::shared_ptr<ControlForceCalculator> createFromConfig(const YAML::Node& config, ros::NodeHandle& node_handle);
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
  void getForceImpl(torch::Tensor& force) override;

 public:
  PotentialFieldMethod(const YAML::Node& config);
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
  static StatePopulator createPopulatorFromString(const Environment& env, const std::string& str);

 public:
  StateProvider(const Environment& env, const std::string& state_pattern);
  torch::Tensor createState();
  ~StateProvider() = default;
  [[nodiscard]] unsigned int getStateDim() const { return state_dim_; };
};

class EpisodeContext {
 private:
  torch::Tensor start_;
  torch::Tensor goal_;
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
  const torch::Tensor& getStart() const { return start_; };
  const torch::Tensor& getGoal() const { return goal_; };
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
  torch::Tensor current_force_;
  double last_calculation_;
  boost::future<torch::Tensor> calculation_future_;
  boost::promise<torch::Tensor> calculation_promise_;
  unsigned int goal_delay_count = 0;
  friend class Visualizer;

  torch::Tensor getAction();
  void calculationRunnable();

 protected:
  const std::string output_dir;
  boost::shared_ptr<StateProvider> state_provider;
  boost::shared_ptr<ros::ServiceClient> training_service_client;

  virtual torch::Tensor getActionInference(torch::Tensor& state) = 0;
  void getForceImpl(torch::Tensor& force) override;

 public:
  ReinforcementLearningAgent(const YAML::Node& config, ros::NodeHandle& node_handle);
  ~ReinforcementLearningAgent() override = default;
};

class DeepQNetworkAgent : public ReinforcementLearningAgent {
 protected:
  torch::Tensor getActionInference(torch::Tensor& state) override;

 public:
  DeepQNetworkAgent(const YAML::Node& config, ros::NodeHandle& node_handle);
};

class MonteCarloAgent : public ReinforcementLearningAgent {
 protected:
  torch::Tensor getActionInference(torch::Tensor& state) override;

 public:
  MonteCarloAgent(const YAML::Node& config, ros::NodeHandle& node_handle);
};
}  // namespace control_force_provider::backend