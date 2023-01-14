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
  const torch::DeviceType device_;
  torch::Tensor ee_position_;
  boost::shared_mutex ee_pos_mtx;
  torch::Tensor ee_rotation_;
  boost::shared_mutex ee_rot_mtx;
  torch::Tensor ee_velocity_;
  boost::shared_mutex ee_vel_mtx;
  torch::Tensor rcm_;
  boost::shared_mutex rcm_mtx;
  torch::Tensor goal_;
  boost::shared_mutex goal_mtx;
  torch::Tensor workspace_bb_origin_;
  boost::shared_mutex bb_or_mtx;
  std::vector<torch::Tensor> ob_positions_;
  boost::shared_mutex ob_pos_mtx;
  std::vector<torch::Tensor> ob_directions_;
  boost::shared_mutex ob_dir_mtx;
  std::vector<torch::Tensor> ob_rotations_;
  boost::shared_mutex ob_rot_mtx;
  std::vector<torch::Tensor> ob_velocities_;
  boost::shared_mutex ob_vel_mtx;
  std::vector<torch::Tensor> ob_rcms_;
  boost::shared_mutex ob_rcm_mtx;
  std::vector<torch::Tensor> points_on_l1_;
  boost::shared_mutex l1_mtx;
  std::vector<torch::Tensor> points_on_l2_;
  boost::shared_mutex l2_mtx;
  std::vector<torch::Tensor> collision_distances_;
  boost::shared_mutex cd_mtx;
  torch::Tensor offset_;
  boost::shared_mutex offset_mtx;
  const torch::Tensor workspace_bb_dims_;
  const torch::Scalar max_force_;
  double start_time_;
  torch::Scalar elapsed_time_;
  std::vector<boost::shared_ptr<Obstacle>> obstacles_;
  boost::shared_ptr<ObstacleLoader> obstacle_loader_;

 public:
  Environment(const YAML::Node& config, int batch_size = 1, torch::DeviceType device = torch::kCPU);
  const torch::Tensor& getEePosition() const;
  boost::shared_lock_guard<boost::shared_mutex> getEePositionLock();
  const torch::Tensor& getEeRotation() const;
  boost::shared_lock_guard<boost::shared_mutex> getEeRotationLock();
  const torch::Tensor& getEeVelocity() const;
  boost::shared_lock_guard<boost::shared_mutex> getEeVelocityLock();
  const torch::Tensor& getRCM() const;
  boost::shared_lock_guard<boost::shared_mutex> getRCMLock();
  void setRCM(const torch::Tensor& rcm);
  const torch::Tensor& getGoal() const;
  boost::shared_lock_guard<boost::shared_mutex> getGoalLock();
  void setGoal(const torch::Tensor& goal);
  const torch::Tensor& getWorkspaceBbOrigin() const;
  boost::shared_lock_guard<boost::shared_mutex> getWorkspaceBbOriginLock();
  const std::vector<torch::Tensor>& getObPositions() const;
  boost::shared_lock_guard<boost::shared_mutex> getObPositionsLock();
  const std::vector<torch::Tensor>& getObDirections() const;
  boost::shared_lock_guard<boost::shared_mutex> getObDirectionsLock();
  const std::vector<torch::Tensor>& getObRotations() const;
  boost::shared_lock_guard<boost::shared_mutex> getObRotationsLock();
  const std::vector<torch::Tensor>& getObVelocities() const;
  boost::shared_lock_guard<boost::shared_mutex> getObVelocitiesLock();
  const std::vector<torch::Tensor>& getObRCMs() const;
  const std::vector<boost::shared_ptr<Obstacle>>& getObstacles() const;
  const boost::shared_ptr<ObstacleLoader>& getObstacleLoader() const;
  boost::shared_lock_guard<boost::shared_mutex> getObRCMsLock();
  const std::vector<torch::Tensor>& getPointsOnL1() const;
  boost::shared_lock_guard<boost::shared_mutex> getPointsOnL1Lock();
  const std::vector<torch::Tensor>& getPointsOnL2() const;
  boost::shared_lock_guard<boost::shared_mutex> getPointsOnL2Lock();
  const std::vector<torch::Tensor>& getCollisionDistances() const;
  boost::shared_lock_guard<boost::shared_mutex> getCollisionDistancesLock();
  const torch::Tensor& getOffset() const;
  boost::shared_lock_guard<boost::shared_mutex> getOffsetLock();
  void setOffset(const torch::Tensor& offset);
  double getStartTime() const;
  void setStartTime(double startTime);
  const torch::Tensor& getWorkspaceBbDims() const;
  double getMaxForce() const;
  double getElapsedTime() const;
  void update(const torch::Tensor& ee_position);
  void clipForce(torch::Tensor& force);
};

class ControlForceCalculator {
 private:
  const inline static double rcm_max_norm = 10;
  bool rcm_available_ = false;
  bool goal_available_ = false;

 protected:
  boost::shared_ptr<Environment> env;
  friend class Visualizer;
  friend class StateProvider;

  virtual void getForceImpl(torch::Tensor& force) = 0;

 public:
  ControlForceCalculator(const YAML::Node& config);
  void getForce(torch::Tensor& force, const torch::Tensor& ee_position_);
  [[nodiscard]] torch::Tensor getRCM() const { return env->getRCM(); }
  void setRCM(const torch::Tensor& rcm_) {
    if (!torch::any(utils::norm(rcm_) > rcm_max_norm).item().toBool()) {
      rcm_available_ = true;
      env->setRCM(rcm_);
    }
  };
  [[nodiscard]] torch::Tensor getGoal() const { return env->getGoal(); }
  void setGoal(const torch::Tensor& goal_) {
    goal_available_ = true;
    env->setGoal(goal_);
    env->setStartTime(Time::now());
  };
  const boost::shared_ptr<Environment>& getEnvironment() const { return env; };
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
    void populate(torch::Tensor& state);
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
  const torch::DeviceType device_;
  torch::Tensor start_;
  torch::Tensor goal_;
  std::vector<boost::shared_ptr<Obstacle>> obstacles_;
  boost::shared_ptr<ObstacleLoader> obstacle_loader_;
  const torch::Tensor start_bb_origin;
  const torch::Tensor start_bb_dims;
  const torch::Tensor goal_bb_origin;
  const torch::Tensor goal_bb_dims;
  const double begin_max_offset_;
  friend class Visualizer;

 public:
  EpisodeContext(std::vector<boost::shared_ptr<Obstacle>> obstacles_, boost::shared_ptr<ObstacleLoader> obstacle_loader, const YAML::Node& config,
                 unsigned int batch_size = 1, torch::DeviceType device = torch::kCPU);
  void generateEpisode(const torch::Tensor& mask);
  void generateEpisode();
  void startEpisode(const torch::Tensor& mask);
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
  boost::shared_ptr<EpisodeContext> episode_context_;
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
  const boost::shared_ptr<EpisodeContext>& getEpisodeContext() { return episode_context_; };
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