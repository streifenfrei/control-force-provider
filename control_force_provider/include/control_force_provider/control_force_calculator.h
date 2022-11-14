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

namespace control_force_provider::backend {
class ControlForceCalculator {
 private:
  const inline static double workspace_bb_stopping_strength = 0.001;
  const inline static double rcm_max_norm = 10;
  bool rcm_available_ = false;
  bool goal_available_ = false;

 protected:
  Eigen::Vector3d workspace_bb_origin_;
  const Eigen::Vector3d workspace_bb_dims_;
  const double max_force_;
  Eigen::Vector4d ee_position;
  Eigen::Vector4d ee_rotation;
  Eigen::Vector4d ee_velocity;
  Eigen::Vector3d rcm;
  Eigen::Vector4d goal;
  ros::Time start_time;
  double elapsed_time = 0;
  std::vector<boost::shared_ptr<Obstacle>> obstacles;
  boost::shared_ptr<ObstacleLoader> obstacle_loader_;
  std::vector<Eigen::Vector4d> ob_positions;
  std::vector<Eigen::Vector4d> ob_rotations;
  std::vector<Eigen::Vector4d> ob_velocities;
  std::vector<Eigen::Vector3d> ob_rcms;
  std::vector<Eigen::Vector3d> points_on_l1_;
  std::vector<Eigen::Vector3d> points_on_l2_;
  Eigen::Vector3d offset_;
  friend class Visualizer;
  friend class StateProvider;

  virtual void getForceImpl(Eigen::Vector4d& force) = 0;

  void updateDistanceVectors();

  void setOffset(Eigen::Vector3d offset);

 public:
  ControlForceCalculator(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, const std::string& data_path);
  void getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position_);
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm; }
  void setRCM(const Eigen::Vector3d& rcm_) {
    if (rcm_.norm() < rcm_max_norm) {
      rcm_available_ = true;
      rcm = rcm_ - offset_;
    }
  };
  [[nodiscard]] const Eigen::Vector4d& getGoal() const { return goal; }
  void setGoal(const Eigen::Vector4d& goal_) {
    goal_available_ = true;
    goal = goal_;
    goal.head(3) -= offset_;
    start_time = ros::Time::now();
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
  void getForceImpl(Eigen::Vector4d& force) override;

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
    std::vector<std::deque<Eigen::VectorXd>> histories_;
    unsigned int stride_index_ = 0;

   public:
    std::vector<const double*> vectors_;
    int length_ = 0;
    unsigned int history_length_ = 1;
    unsigned int history_stride_ = 0;
    StatePopulator() = default;
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

class EpisodeContext {
 private:
  Eigen::Vector4d start_;
  Eigen::Vector4d goal_;
  std::vector<boost::shared_ptr<Obstacle>>& obstacles_;
  boost::shared_ptr<ObstacleLoader>& obstacle_loader_;
  boost::random::mt19937 rng_;
  const Eigen::Vector3d start_bb_origin;
  const Eigen::Vector3d start_bb_dims;
  const Eigen::Vector3d goal_bb_origin;
  const Eigen::Vector3d goal_bb_dims;
  double begin_max_offset_;
  friend class Visualizer;

 public:
  EpisodeContext(std::vector<boost::shared_ptr<Obstacle>>& obstacles_, boost::shared_ptr<ObstacleLoader>& obstacle_loader, const YAML::Node& config);
  void generateEpisode();
  void startEpisode();
  const Eigen::Vector4d& getStart() const { return start_; };
  const Eigen::Vector4d& getGoal() const { return goal_; };
};

class ReinforcementLearningAgent : public ControlForceCalculator {
 private:
  const inline static double transition_smoothness = 0.001;
  const inline static unsigned int goal_delay = 10;
  const bool train;
  const ros::Duration interval_duration_;
  const double goal_reached_threshold_distance_;
  const bool rcm_origin_;
  EpisodeContext episode_context_;
  bool initializing_episode = true;
  Eigen::Vector4d current_force_;
  ros::Time last_calculation_;
  boost::future<Eigen::Vector4d> calculation_future_;
  boost::promise<Eigen::Vector4d> calculation_promise_;
  unsigned int goal_delay_count = 0;
  friend class Visualizer;

  Eigen::Vector4d getAction();
  void calculationRunnable();

 protected:
  const std::string output_dir;
  boost::shared_ptr<StateProvider> state_provider;
  boost::shared_ptr<ros::ServiceClient> training_service_client;

  virtual torch::Tensor getActionInference(torch::Tensor& state) = 0;
  void getForceImpl(Eigen::Vector4d& force) override;

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
}  // namespace control_force_provider::backend