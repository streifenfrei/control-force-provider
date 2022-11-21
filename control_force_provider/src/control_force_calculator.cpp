#include "control_force_provider/control_force_calculator.h"

#include <boost/algorithm/clamp.hpp>
#include <boost/array.hpp>
#include <boost/chrono.hpp>
#include <string>
#include <utility>

#include "control_force_provider/utils.h"

using namespace Eigen;
namespace control_force_provider::backend {
ControlForceCalculator::ControlForceCalculator(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, const std::string& data_path)
    : obstacles(std::move(obstacles_)),
      rcm(Vector3d::Zero()),
      workspace_bb_origin_(utils::vectorFromList(utils::getConfigValue<double>(config, "workspace_bb"), 0)),
      workspace_bb_dims_(utils::vectorFromList(utils::getConfigValue<double>(config, "workspace_bb"), 3)),
      max_force_(utils::getConfigValue<double>(config, "max_force")[0]),
      offset_(Vector3d::Zero()) {
  std::vector<boost::shared_ptr<FramesObstacle>> frames_obstacles;
  int reference_obstacle = -1;
  for (auto& obstacle : obstacles) {
    ob_rcms.push_back(obstacle->getRCM());
    ob_positions.emplace_back();
    ob_rotations.emplace_back();
    ob_velocities.emplace_back();
    points_on_l1_.emplace_back();
    points_on_l2_.emplace_back();
    boost::shared_ptr<FramesObstacle> frames_obstacle = boost::dynamic_pointer_cast<FramesObstacle>(obstacle);
    if (frames_obstacle) {
      if (frames_obstacle->getRCM() != Vector3d::Zero()) {
        if (reference_obstacle == -1)
          reference_obstacle = frames_obstacles.size();
        else
          throw utils::ConfigError("Found more than one reference RCM for the obstacles");
      }
      frames_obstacles.push_back(frames_obstacle);
    }
  }
  obstacle_loader_ = boost::make_shared<ObstacleLoader>(frames_obstacles, data_path, reference_obstacle);
}

void ControlForceCalculator::getForce(Vector4d& force, const Vector4d& ee_position_) {
  Vector4d ee_position_off = ee_position_;
  ee_position_off.head(3) -= offset_;
  if (!goal_available_) setGoal(ee_position);
  if (rcm_available_) {
    ee_velocity = ee_position_off - ee_position;
    ee_position = ee_position_off;
    Quaterniond ee_rot = utils::zRotation(rcm - offset_, ee_position.head(3));
    ee_rotation = {ee_rot.x(), ee_rot.y(), ee_rot.z(), ee_rot.w()};
    elapsed_time = Time::now() - start_time;
    for (size_t i = 0; i < obstacles.size(); i++) {
      Vector4d new_ob_position = obstacles[i]->getPosition();
      new_ob_position.head(3) -= offset_;
      ob_velocities[i] = new_ob_position - ob_positions[i];
      ob_positions[i] = new_ob_position;
      ob_rcms[i] = obstacles[i]->getRCM() - offset_;
      Quaterniond ob_rot = utils::zRotation(ob_rcms[i], ob_positions[i].head(3));
      ob_rotations[i] = {ob_rot.x(), ob_rot.y(), ob_rot.z(), ob_rot.w()};
    }
    updateDistanceVectors();
    getForceImpl(force);
  }
  // clip force to workspace bb
  Vector4d next_pos = ee_position + force;
  double distance_to_wall = INFINITY;
  double next_distance_to_wall = INFINITY;
  for (size_t i = 0; i < 3; i++) {
    distance_to_wall = std::min(distance_to_wall, ee_position[i] - workspace_bb_origin_[i]);
    distance_to_wall = std::min(distance_to_wall, workspace_bb_origin_[i] + workspace_bb_dims_[i] - ee_position[i]);
    next_distance_to_wall = std::min(next_distance_to_wall, next_pos[i] - workspace_bb_origin_[i]);
    next_distance_to_wall = std::min(next_distance_to_wall, workspace_bb_origin_[i] + workspace_bb_dims_[i] - next_pos[i]);
    next_pos[i] = boost::algorithm::clamp(next_pos[i], workspace_bb_origin_[i], workspace_bb_origin_[i] + workspace_bb_dims_[i]);
  }
  if (next_distance_to_wall < distance_to_wall) {
    force = next_pos - ee_position;
    if (distance_to_wall > 0) {
      double max_magnitude = distance_to_wall * workspace_bb_stopping_strength;
      double magnitude = force.norm();
      if (magnitude > max_magnitude) force = force / magnitude * max_magnitude;
    }
  }
}

void ControlForceCalculator::updateDistanceVectors() {
  const Vector3d& goal3d = goal.head(3);
  const Vector3d& ee_position3d = ee_position.head(3);
  const Vector3d& a1 = rcm;
  Vector3d b1 = ee_position3d - a1;
  for (size_t i = 0; i < obstacles.size(); i++) {
    double t, s = 0;
    const Vector3d& a2 = ob_rcms[i];
    Vector3d b2 = ob_positions[i].head(3) - a2;
    utils::shortestLine(a1, b1, a2, b2, t, s);
    t = boost::algorithm::clamp(t, 0, 1);
    s = boost::algorithm::clamp(s, 0, 1);
    points_on_l1_[i] = (a1 + t * b1);
    points_on_l2_[i] = (a2 + s * b2);
  }
}

void ControlForceCalculator::setOffset(Vector3d offset) {
  Vector3d translation = offset_ - offset;
  workspace_bb_origin_ += translation;
  rcm += translation;
  goal.head(3) += translation;
  offset_ = offset;
}

PotentialFieldMethod::PotentialFieldMethod(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, const std::string& data_path)
    : ControlForceCalculator(std::move(obstacles_), config, data_path),
      attraction_strength_(utils::getConfigValue<double>(config["pfm"], "attraction_strength")[0]),
      attraction_distance_(utils::getConfigValue<double>(config["pfm"], "attraction_distance")[0]),
      repulsion_strength_(utils::getConfigValue<double>(config["pfm"], "repulsion_strength")[0]),
      repulsion_distance_(utils::getConfigValue<double>(config["pfm"], "repulsion_distance")[0]),
      z_translation_strength_(utils::getConfigValue<double>(config["pfm"], "z_translation_strength")[0]),
      min_rcm_distance_(utils::getConfigValue<double>(config["pfm"], "min_rcm_distance")[0]) {
  for (auto& obstacle : obstacles) {
    points_on_l1_.emplace_back();
    points_on_l2_.emplace_back();
  }
}

void PotentialFieldMethod::getForceImpl(Vector4d& force) {
  const Vector3d& goal3d = goal.head(3);
  const Vector3d& ee_position3d = ee_position.head(3);
  // attractive vector
  Vector3d attractive_vector = goal3d - ee_position3d;
  double ee_to_goal_distance = attractive_vector.norm();
  double smoothing_factor = 1;
  if (ee_to_goal_distance > attraction_distance_) {
    attractive_vector.normalize();  // use linear potential when far away from goal
    smoothing_factor = 0.1;
  }
  attractive_vector *= attraction_strength_ * smoothing_factor;

  // repulsive vector
  Vector3d repulsive_vector = Vector3d::Zero();
  const Vector3d& a1 = rcm;
  Vector3d b1 = ee_position3d - a1;
  double l1_length = b1.norm();
  double min_l2_to_l1_distance = repulsion_distance_;
  for (size_t i = 0; i < obstacles.size(); i++) {
    double t = (points_on_l1_[i] - rcm).norm() / l1_length;
    Vector3d l2_to_l1 = points_on_l1_[i] - points_on_l2_[i];
    double l2_to_l1_distance = l2_to_l1.norm();
    if (l2_to_l1_distance < repulsion_distance_) {
      repulsive_vector += (repulsion_strength_ / l2_to_l1_distance - repulsion_strength_ / repulsion_distance_) / (l2_to_l1_distance * l2_to_l1_distance) *
                          l2_to_l1.normalized() * t * l1_length;
      if (l2_to_l1_distance < min_l2_to_l1_distance) min_l2_to_l1_distance = l2_to_l1_distance;
    }
  }
  if (min_l2_to_l1_distance < repulsion_distance_) {
    // avoid positive z-translation
    Vector3d l1_new = ee_position3d + repulsive_vector - a1;
    double l1_new_length = l1_new.norm();
    if (l1_length - l1_new_length < 0) {
      l1_new *= l1_length / l1_new_length;
      repulsive_vector = a1 + l1_new - ee_position3d;
    }
    // add negative z-translation
    repulsive_vector -= (z_translation_strength_ / min_l2_to_l1_distance - z_translation_strength_ / repulsion_distance_) /
                        (min_l2_to_l1_distance * min_l2_to_l1_distance) * (b1 / l1_length);
  }
  if ((ee_position3d + repulsive_vector - a1).norm() < min_rcm_distance_)  // prevent pushing the end effector too close to the RCM
    repulsive_vector = {0, 0, 0};
  Vector3d force3d = attractive_vector + repulsive_vector;
  force = {force3d[0], force3d[1], force3d[2], 0};
}

StateProvider::StatePopulator StateProvider::createPopulatorFromString(const ControlForceCalculator& cfc, const std::string& str) {
  std::string id = str.substr(0, 3);
  std::string args = str.substr(4, str.length() - 5);
  StatePopulator populator;
  populator.length_ = 3;
  if (id == "ree") {
    populator.vectors_.push_back(cfc.ee_position.data());
  } else if (id == "rve") {
    populator.vectors_.push_back(cfc.ee_velocity.data());
  } else if (id == "rro") {
    populator.length_ = 4;
    populator.vectors_.push_back(cfc.ee_rotation.data());
  } else if (id == "rpp") {
    populator.vectors_.push_back(cfc.rcm.data());
  } else if (id == "oee") {
    for (auto& pos : cfc.ob_positions) populator.vectors_.push_back(pos.data());
  } else if (id == "ove") {
    for (auto& vel : cfc.ob_velocities) populator.vectors_.push_back(vel.data());
  } else if (id == "oro") {
    populator.length_ = 4;
    for (auto& rot : cfc.ob_rotations) populator.vectors_.push_back(rot.data());
  } else if (id == "opp") {
    for (auto& rcm : cfc.ob_rcms) populator.vectors_.push_back(rcm.data());
  } else if (id == "gol") {
    populator.vectors_.push_back(cfc.goal.data());
  } else if (id == "tim") {
    populator.length_ = 1;
    populator.vectors_.push_back(&cfc.elapsed_time);
  } else {
    ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Unknown id in state pattern: " << id);
  }
  // parse arguments
  for (auto& arg : utils::regexFindAll(arg_regex, args)) {
    std::string arg_id = arg.substr(0, 1);
    unsigned int value = std::stoi(arg.substr(1, arg.length() - 1));
    if (arg_id == "h")
      populator.history_length_ = value;
    else if (arg_id == "s")
      populator.history_stride_ = value;
    else
      ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Unknown argument " << arg << " for id " << id << " in state pattern");
  }
  return populator;
}

StateProvider::StateProvider(const ControlForceCalculator& cfc, const std::string& state_pattern) {
  for (auto& str : utils::regexFindAll(pattern_regex, state_pattern)) {
    state_populators_.push_back(createPopulatorFromString(cfc, str));
    state_dim_ += state_populators_.back().getDim();
  }
}

void StateProvider::StatePopulator::populate(torch::Tensor& state, int& index) {
  auto state_accessor = state.accessor<double, 1>();
  for (size_t i = 0; i < vectors_.size(); i++) {
    const double* vector = vectors_[i];
    if (histories_.size() <= i) histories_.emplace_back();
    std::deque<VectorXd>& history = histories_[i];
    if (stride_index_ == 0) {
      VectorXd eigen_vector(length_);
      for (size_t j = 0; j < length_; j++) eigen_vector[j] = vector[j];
      do history.push_back(eigen_vector);
      while (history.size() <= history_length_);
      history.pop_front();
    }
    for (auto& vec : history) {
      for (size_t j = 0; j < length_; j++) state_accessor[index++] = vec[j];
    }
  }
  stride_index_ = (stride_index_ - 1) % (history_stride_ + 1);
}

torch::Tensor StateProvider::createState() {
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor state = torch::empty(state_dim_, options);
  int index = 0;
  for (auto& pop : state_populators_) pop.populate(state, index);
  return state;
}

EpisodeContext::EpisodeContext(std::vector<boost::shared_ptr<Obstacle>>& obstacles, boost::shared_ptr<ObstacleLoader>& obstacle_loader,
                               const YAML::Node& config)
    : obstacles_(obstacles),
      obstacle_loader_(obstacle_loader),
      begin_max_offset_(utils::getConfigValue<double>(config, "begin_max_offset")[0]),
      start_bb_origin(utils::vectorFromList(utils::getConfigValue<double>(config, "start_bb"), 0)),
      start_bb_dims(utils::vectorFromList(utils::getConfigValue<double>(config, "start_bb"), 3)),
      goal_bb_origin(utils::vectorFromList(utils::getConfigValue<double>(config, "goal_bb"), 0)),
      goal_bb_dims(utils::vectorFromList(utils::getConfigValue<double>(config, "goal_bb"), 3)) {}

void EpisodeContext::generateEpisode() {
  for (size_t i = 0; i < 3; i++) {
    start_[i] = boost::random::uniform_real_distribution<>(start_bb_origin[i], start_bb_origin[i] + start_bb_dims[i])(rng_);
    goal_[i] = boost::random::uniform_real_distribution<>(goal_bb_origin[i], goal_bb_origin[i] + goal_bb_dims[i])(rng_);
  }
  obstacle_loader_->loadNext();
}

void EpisodeContext::startEpisode() {
  double offset = boost::random::uniform_real_distribution<>(0, begin_max_offset_)(rng_);
  for (auto& obstacle : obstacles_) obstacle->reset(offset);
}

ReinforcementLearningAgent::ReinforcementLearningAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config,
                                                       ros::NodeHandle& node_handle, const std::string& data_path)
    : ControlForceCalculator(std::move(obstacles_), config, data_path),
      interval_duration_(utils::getConfigValue<double>(config["rl"], "interval_duration")[0] * 10e-4),
      goal_reached_threshold_distance_(utils::getConfigValue<double>(config["rl"], "goal_reached_threshold_distance")[0]),
      episode_context_(obstacles, obstacle_loader_, config["rl"]),
      train(utils::getConfigValue<bool>(config["rl"], "train")[0]),
      rcm_origin_(utils::getConfigValue<bool>(config["rl"], "rcm_origin")[0]),
      output_dir(utils::getConfigValue<std::string>(config["rl"], "output_directory")[0]),
      last_calculation_(Time::now()) {
  if (rcm_origin_) setOffset(rcm);
  state_provider = boost::make_shared<StateProvider>(*this, utils::getConfigValue<std::string>(config["rl"], "state_pattern")[0]);
  calculation_future_ = calculation_promise_.get_future();
  calculation_promise_.set_value(Vector4d::Zero());
  if (train) {
    training_service_client = boost::make_shared<ros::ServiceClient>(node_handle.serviceClient<control_force_provider_msgs::UpdateNetwork>("update_network"));
  }
  episode_context_.generateEpisode();
  setGoal(episode_context_.getStart());
  initializing_episode = true;
}

Vector4d ReinforcementLearningAgent::getAction() {
  if (train) {
    if (training_service_client->exists()) {
      torch::Tensor state_tensor = state_provider->createState();
      auto state_tensor_accessor = state_tensor.accessor<double, 1>();
      std::vector<double> state_vector;
      for (size_t i = 0; i < state_provider->getStateDim(); i++) state_vector.push_back(state_tensor_accessor[i]);
      control_force_provider_msgs::UpdateNetwork srv;
      srv.request.state = state_vector;
      for (size_t i = 0; i < 4; i++) srv.request.robot_position[i] = ee_position[i];
      for (size_t i = 0; i < 4; i++) srv.request.robot_velocity[i] = ee_velocity[i];
      for (size_t i = 0; i < 3; i++) srv.request.robot_rcm[i] = rcm[i];
      for (auto& ob_position : ob_positions)
        for (size_t i = 0; i < ob_position.size(); i++) srv.request.obstacles_positions.push_back(ob_position[i]);
      for (auto& ob_velocity : ob_velocities)
        for (size_t i = 0; i < ob_velocity.size(); i++) srv.request.obstacles_velocities.push_back(ob_velocity[i]);
      for (auto& ob_rcm : ob_rcms)
        for (size_t i = 0; i < ob_rcm.size(); i++) srv.request.obstacles_rcms.push_back(ob_rcm[i]);
      for (auto& point : points_on_l1_)
        for (size_t i = 0; i < point.size(); i++) srv.request.points_on_l1.push_back(point[i]);
      for (auto& point : points_on_l2_)
        for (size_t i = 0; i < point.size(); i++) srv.request.points_on_l2.push_back(point[i]);
      for (size_t i = 0; i < 4; i++) srv.request.goal[i] = goal[i];
      for (size_t i = 0; i < 3; i++) srv.request.workspace_bb_origin[i] = workspace_bb_origin_[i];
      for (size_t i = 0; i < 3; i++) srv.request.workspace_bb_dims[i] = workspace_bb_dims_[i];
      srv.request.elapsed_time = elapsed_time;
      if (!training_service_client->call(srv)) ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Failed to call training service.");
      return Vector4d(srv.response.action.data());
    } else
      return Vector4d::Zero();
  } else {
    // TODO: extrapolate current state during inference
    // torch::Tensor state = getActionInference(...);
    return Vector4d::Zero();
  }
}

void ReinforcementLearningAgent::calculationRunnable() { calculation_promise_.set_value(getAction()); }

void ReinforcementLearningAgent::getForceImpl(Vector4d& force) {
  if (rcm_origin_ && rcm != Vector3d::Zero()) {
    setOffset(rcm);
    force = current_force_;
    return;
  }
  double now = Time::now();
  if (now - last_calculation_ > interval_duration_) {
    if (train) {
      Vector4d goal_vector = goal - ee_position;
      goal_vector[3] = 0;
      if (goal_vector.norm() < goal_reached_threshold_distance_) {
        if (initializing_episode) {
          episode_context_.startEpisode();
          setGoal(episode_context_.getGoal());
          initializing_episode = false;
        } else {
          goal_delay_count++;
          if (goal_delay_count >= goal_delay) {
            episode_context_.generateEpisode();
            setGoal(episode_context_.getStart());
            initializing_episode = true;
            goal_delay_count = 0;
          }
        }
      } else {
        goal_delay_count = 0;
      }
      current_force_ = initializing_episode ? goal_vector * transition_smoothness : getAction();
    } else {
      if (calculation_future_.wait_for(boost::chrono::seconds(0)) == boost::future_status::ready) {
        current_force_ = calculation_future_.get();
        calculation_promise_ = {};
        calculation_future_ = calculation_promise_.get_future();
        boost::thread calculation_thread{&ReinforcementLearningAgent::calculationRunnable, this};
      } else {
        ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "The RL agent exceeded the time limit!");
      }
    }
    last_calculation_ = now;
    if (!current_force_.allFinite() || current_force_.hasNaN()) {
      ROS_WARN_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "The force vector contains Infs or NaNs.");
      for (size_t i = 0; i < 4; i++) {
        double value = current_force_[i];
        if (!std::isfinite(value) || std::isnan(value)) current_force_[i] = 0;
      }
    }
  }
  force = current_force_;
}

DeepQNetworkAgent::DeepQNetworkAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle,
                                     const std::string& data_path)
    : ReinforcementLearningAgent(std::move(obstacles_), config, node_handle, data_path) {}

torch::Tensor DeepQNetworkAgent::getActionInference(torch::Tensor& state) {}
}  // namespace control_force_provider::backend