#include "control_force_provider/control_force_calculator.h"

#include <boost/algorithm/clamp.hpp>
#include <boost/array.hpp>
#include <boost/chrono.hpp>
#include <string>
#include <utility>

#include "control_force_provider/utils.h"

using namespace control_force_provider::utils;
using namespace Eigen;
namespace control_force_provider::backend {

boost::shared_ptr<ControlForceCalculator> ControlForceCalculator::createFromConfig(const YAML::Node& config, ros::NodeHandle& node_handle) {
  boost::shared_ptr<ControlForceCalculator> control_force_calculator;
  std::string calculator_type = getConfigValue<std::string>(config, "algorithm")[0];
  if (calculator_type == "pfm") {
    control_force_calculator = boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<PotentialFieldMethod>(config));
  } else if (calculator_type == "rl") {
    std::string rl_type = getConfigValue<std::string>((config)["rl"], "type")[0];
    if (rl_type == "dqn") {
      control_force_calculator = boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<DeepQNetworkAgent>(config, node_handle));
    } else if (rl_type == "mc") {
      control_force_calculator = boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<MonteCarloAgent>(config, node_handle));
    } else
      throw ConfigError("Unknown RL type '" + rl_type + "'");
  } else
    throw ConfigError("Unknown calculator type '" + calculator_type + "'");
  return control_force_calculator;
}

Environment::Environment(const YAML::Node& config)
    : ee_position(torch::zeros({1, 3}, utils::getTensorOptions())),
      ee_rotation(torch::zeros({1, 4}, utils::getTensorOptions())),
      ee_velocity(torch::zeros({1, 3}, utils::getTensorOptions())),
      rcm(torch::zeros(3, utils::getTensorOptions())),
      goal(torch::zeros({1, 3}, utils::getTensorOptions())),
      start_time(0),
      elapsed_time(0),
      workspace_bb_origin(utils::createTensor(utils::getConfigValue<double>(config, "workspace_bb"), 0, 3)),
      workspace_bb_dims(utils::createTensor(utils::getConfigValue<double>(config, "workspace_bb"), 3, 6)),
      max_force(utils::getConfigValue<double>(config, "max_force")[0]),
      offset(torch::zeros(3, utils::getTensorOptions())) {
  std::string data_path;
  obstacles = Obstacle::createFromConfig(config, data_path);
  std::vector<boost::shared_ptr<FramesObstacle>> frames_obstacles;
  int reference_obstacle = -1;
  for (auto& obstacle : obstacles) {
    ob_rcms.push_back(obstacle->getRCM());
    ob_positions.push_back(torch::zeros({1, 3}, utils::getTensorOptions()));
    ob_rotations.push_back(torch::zeros({1, 4}, utils::getTensorOptions()));
    ob_velocities.push_back(torch::zeros({1, 3}, utils::getTensorOptions()));
    points_on_l1_.push_back(torch::zeros({1, 3}, utils::getTensorOptions()));
    points_on_l2_.push_back(torch::zeros({1, 3}, utils::getTensorOptions()));
    boost::shared_ptr<FramesObstacle> frames_obstacle = boost::dynamic_pointer_cast<FramesObstacle>(obstacle);
    if (frames_obstacle) {
      if (frames_obstacle->getRCM().equal(torch::zeros(3, utils::getTensorOptions()))) {
        if (reference_obstacle == -1)
          reference_obstacle = frames_obstacles.size();
        else
          throw utils::ConfigError("Found more than one reference RCM for the obstacles");
      }
      frames_obstacles.push_back(frames_obstacle);
    }
  }
  obstacle_loader = boost::make_shared<ObstacleLoader>(frames_obstacles, data_path, reference_obstacle);
}

void Environment::update(const torch::Tensor& ee_position_) {
  torch::Tensor ee_position_off = ee_position_ - offset;
  ee_position = ee_position_off;
  ee_rotation = utils::zRotation(rcm - offset, ee_position);
  elapsed_time = Time::now() - start_time;
  for (size_t i = 0; i < obstacles.size(); i++) {
    torch::Tensor new_ob_position = obstacles[i]->getPosition();
    new_ob_position -= offset;
    ob_velocities[i] = new_ob_position - ob_positions[i];
    ob_positions[i] = new_ob_position;
    ob_rcms[i] = obstacles[i]->getRCM() - offset;
    ob_rotations[i] = utils::zRotation(ob_rcms[i], ob_positions[i]);
  }
  torch::Tensor a1 = rcm;
  torch::Tensor b1 = ee_position - a1;
  for (size_t i = 0; i < obstacles.size(); i++) {
    torch::Tensor t, s = torch::empty(ee_position.size(0), utils::getTensorOptions());
    const torch::Tensor& a2 = ob_rcms[i];
    torch::Tensor b2 = ob_positions[i] - a2;
    utils::shortestLine(a1, b1, a2, b2, t, s);
    t = torch::clamp(t, 0, 1);
    s = torch::clamp(s, 0, 1);
    points_on_l1_[i] = a1 + t * b1;
    points_on_l2_[i] = a2 + s * b2;
  }
}

void Environment::clipForce(torch::Tensor& force_) {
  torch::Tensor next_pos = ee_position + force_;
  torch::Tensor distance_to_wall = torch::full_like(force_, INFINITY);
  torch::Tensor next_distance_to_wall = torch::full_like(force_, INFINITY);
  distance_to_wall = torch::min(distance_to_wall, ee_position - workspace_bb_origin);
  distance_to_wall = torch::min(distance_to_wall, workspace_bb_origin + workspace_bb_dims - ee_position);
  distance_to_wall = std::get<0>(torch::min(distance_to_wall, -1));
  next_distance_to_wall = torch::min(next_distance_to_wall, next_pos - workspace_bb_origin);
  next_distance_to_wall = torch::min(next_distance_to_wall, workspace_bb_origin + workspace_bb_dims - next_pos);
  next_distance_to_wall = std::get<0>(torch::min(next_distance_to_wall, -1));
  next_pos = torch::clamp(next_pos, workspace_bb_origin, workspace_bb_origin + workspace_bb_dims);
  torch::Tensor mask = next_distance_to_wall < distance_to_wall;
  if (torch::any(mask).item().toBool()) {
    force_ = torch::where(mask, next_pos - ee_position, force_);
    mask = torch::logical_and(mask, distance_to_wall > 0);
    if (torch::any(mask).item().toBool()) {
      torch::Tensor max_magnitude = distance_to_wall * workspace_bb_stopping_strength;
      torch::Tensor magnitude = utils::norm(force_);
      mask = torch::logical_and(mask, magnitude > max_magnitude);
      if (torch::any(mask).item().toBool()) force_ = torch::where(mask, force_ / magnitude * max_magnitude, force_);
    }
  }
}

void Environment::setOffset(const torch::Tensor& offset_) {
  // TODO fix it
  torch::Tensor translation = offset - offset_;
  workspace_bb_origin += offset_;
  rcm += translation;
  goal += translation;
  offset = offset_;
}
ControlForceCalculator::ControlForceCalculator(const YAML::Node& config) : env(config) {}

void ControlForceCalculator::getForce(torch::Tensor& force, const torch::Tensor& ee_position_) {
  // TODO: use tensors only
  if (!goal_available_) setGoal(ee_position_ - env.offset);
  if (rcm_available_) {
    env.update(ee_position_);
    getForceImpl(force);
  }
}

PotentialFieldMethod::PotentialFieldMethod(const YAML::Node& config)
    : ControlForceCalculator(config),
      attraction_strength_(utils::getConfigValue<double>(config["pfm"], "attraction_strength")[0]),
      attraction_distance_(utils::getConfigValue<double>(config["pfm"], "attraction_distance")[0]),
      repulsion_strength_(utils::getConfigValue<double>(config["pfm"], "repulsion_strength")[0]),
      repulsion_distance_(utils::getConfigValue<double>(config["pfm"], "repulsion_distance")[0]),
      z_translation_strength_(utils::getConfigValue<double>(config["pfm"], "z_translation_strength")[0]),
      min_rcm_distance_(utils::getConfigValue<double>(config["pfm"], "min_rcm_distance")[0]) {}

void PotentialFieldMethod::getForceImpl(torch::Tensor& force) {
  // attractive vector
  // TODO: replace with tensors fully
  Vector3d attractive_vector = utils::tensorToVector(env.goal - env.ee_position);
  double ee_to_goal_distance = attractive_vector.norm();
  double smoothing_factor = 1;
  if (ee_to_goal_distance > attraction_distance_) {
    attractive_vector.normalize();  // use linear potential when far away from goal
    smoothing_factor = 0.1;
  }
  attractive_vector *= attraction_strength_ * smoothing_factor;

  // repulsive vector
  Vector3d repulsive_vector = Vector3d::Zero();
  Vector3d a1 = utils::tensorToVector(env.rcm);
  Vector3d b1 = utils::tensorToVector(env.ee_position) - a1;
  double l1_length = b1.norm();
  double min_l2_to_l1_distance = repulsion_distance_;
  for (size_t i = 0; i < env.obstacles.size(); i++) {
    double t = utils::tensorToVector(env.points_on_l1_[i] - env.rcm).norm() / l1_length;
    Vector3d l2_to_l1 = utils::tensorToVector(env.points_on_l1_[i] - env.points_on_l2_[i]);
    double l2_to_l1_distance = l2_to_l1.norm();
    if (l2_to_l1_distance < repulsion_distance_) {
      repulsive_vector += (repulsion_strength_ / l2_to_l1_distance - repulsion_strength_ / repulsion_distance_) / (l2_to_l1_distance * l2_to_l1_distance) *
                          l2_to_l1.normalized() * t * l1_length;
      if (l2_to_l1_distance < min_l2_to_l1_distance) min_l2_to_l1_distance = l2_to_l1_distance;
    }
  }
  if (min_l2_to_l1_distance < repulsion_distance_) {
    // avoid positive z-translation
    Vector3d l1_new = utils::tensorToVector(env.ee_position) + repulsive_vector - a1;
    double l1_new_length = l1_new.norm();
    if (l1_length - l1_new_length < 0) {
      l1_new *= l1_length / l1_new_length;
      repulsive_vector = a1 + l1_new - utils::tensorToVector(env.ee_position);
    }
    // add negative z-translation
    repulsive_vector -= (z_translation_strength_ / min_l2_to_l1_distance - z_translation_strength_ / repulsion_distance_) /
                        (min_l2_to_l1_distance * min_l2_to_l1_distance) * (b1 / l1_length);
  }
  if ((utils::tensorToVector(env.ee_position) + repulsive_vector - a1).norm() < min_rcm_distance_)  // prevent pushing the end effector too close to the RCM
    repulsive_vector = {0, 0, 0};
  force = utils::vectorToTensor(attractive_vector + repulsive_vector);
}

StateProvider::StatePopulator StateProvider::createPopulatorFromString(const Environment& env, const std::string& str) {
  std::string id = str.substr(0, 3);
  std::string args = str.substr(4, str.length() - 5);
  StatePopulator populator;
  populator.length_ = 3;
  if (id == "ree") {
    populator.tensors_.push_back(&env.ee_position);
  } else if (id == "rve") {
    populator.tensors_.push_back(&env.ee_velocity);
  } else if (id == "rro") {
    populator.length_ = 4;
    populator.tensors_.push_back(&env.ee_rotation);
  } else if (id == "rpp") {
    populator.tensors_.push_back(&env.rcm);
  } else if (id == "oee") {
    for (auto& pos : env.ob_positions) populator.tensors_.push_back(&pos);
  } else if (id == "ove") {
    for (auto& vel : env.ob_velocities) populator.tensors_.push_back(&vel);
  } else if (id == "oro") {
    populator.length_ = 4;
    for (auto& rot : env.ob_rotations) populator.tensors_.push_back(&rot);
  } else if (id == "opp") {
    for (auto& rcm : env.ob_rcms) populator.tensors_.push_back(&rcm);
  } else if (id == "gol") {
    populator.tensors_.push_back(&env.goal);
    //} else if (id == "tim") {
    //  populator.length_ = 1;
    //  populator.tensors_.push_back(&cfc.elapsed_time);

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

StateProvider::StateProvider(const Environment& env, const std::string& state_pattern) {
  for (auto& str : utils::regexFindAll(pattern_regex, state_pattern)) {
    state_populators_.push_back(createPopulatorFromString(env, str));
    state_dim_ += state_populators_.back().getDim();
  }
}

void StateProvider::StatePopulator::populate(torch::Tensor& state, int& index) {
  auto state_accessor = state.accessor<double, 1>();
  for (size_t i = 0; i < tensors_.size(); i++) {
    const torch::Tensor* tensor = tensors_[i];
    if (histories_.size() <= i) histories_.emplace_back();
    std::deque<torch::Tensor>& history = histories_[i];
    if (stride_index_ == 0) {
      do history.push_back(*tensor);
      while (history.size() <= history_length_);
      history.pop_front();
    }
    for (auto& ten : history) {
      auto acc = ten.accessor<double, 1>();
      for (size_t j = 0; j < length_; j++) state_accessor[index++] = acc[j];
    }
  }
  stride_index_ = (stride_index_ - 1) % (history_stride_ + 1);
}

torch::Tensor StateProvider::createState() {
  torch::Tensor state = torch::empty(state_dim_, utils::getTensorOptions());
  int index = 0;
  for (auto& pop : state_populators_) pop.populate(state, index);
  return state;
}

EpisodeContext::EpisodeContext(std::vector<boost::shared_ptr<Obstacle>>& obstacles, boost::shared_ptr<ObstacleLoader>& obstacle_loader,
                               const YAML::Node& config)
    : obstacles_(obstacles),
      obstacle_loader_(obstacle_loader),
      begin_max_offset_(utils::getConfigValue<double>(config, "begin_max_offset")[0]),
      start_(torch::zeros(3, utils::getTensorOptions())),
      goal_(torch::zeros(3, utils::getTensorOptions())),
      start_bb_origin(utils::createTensor(utils ::getConfigValue<double>(config, "start_bb"), 0, 3)),
      start_bb_dims(utils::createTensor(utils::getConfigValue<double>(config, "start_bb"), 3, 6)),
      goal_bb_origin(utils::createTensor(utils::getConfigValue<double>(config, "goal_bb"), 0, 3)),
      goal_bb_dims(utils::createTensor(utils::getConfigValue<double>(config, "goal_bb"), 3, 6)) {}

void EpisodeContext::generateEpisode() {
  auto start_bb_origin_acc = start_bb_origin.accessor<double, 1>();
  auto start_bb_dims_acc = start_bb_dims.accessor<double, 1>();
  auto goal_bb_origin_acc = goal_bb_origin.accessor<double, 1>();
  auto goal_bb_dims_acc = goal_bb_dims.accessor<double, 1>();
  for (size_t i = 0; i < 3; i++) {
    start_[i] = boost::random::uniform_real_distribution<>(start_bb_origin_acc[i], start_bb_origin_acc[i] + start_bb_dims_acc[i])(rng_);
    goal_[i] = boost::random::uniform_real_distribution<>(goal_bb_origin_acc[i], goal_bb_origin_acc[i] + goal_bb_dims_acc[i])(rng_);
  }
  obstacle_loader_->loadNext();
}

void EpisodeContext::startEpisode() {
  torch::Tensor offset = begin_max_offset_ * torch::rand(1, utils::getTensorOptions());
  for (auto& obstacle : obstacles_) obstacle->reset(offset);
}

ReinforcementLearningAgent::ReinforcementLearningAgent(const YAML::Node& config, ros::NodeHandle& node_handle)
    : ControlForceCalculator(config),
      interval_duration_(utils::getConfigValue<double>(config["rl"], "interval_duration")[0] * 10e-4),
      goal_reached_threshold_distance_(utils::getConfigValue<double>(config["rl"], "goal_reached_threshold_distance")[0]),
      episode_context_(env.obstacles, env.obstacle_loader, config["rl"]),
      train(utils::getConfigValue<bool>(config["rl"], "train")[0]),
      rcm_origin_(utils::getConfigValue<bool>(config["rl"], "rcm_origin")[0]),
      output_dir(utils::getConfigValue<std::string>(config["rl"], "output_directory")[0]),
      current_force_(torch::zeros(3, utils::getTensorOptions())),
      last_calculation_(Time::now()) {
  if (rcm_origin_) env.setOffset(env.rcm);
  state_provider = boost::make_shared<StateProvider>(env, utils::getConfigValue<std::string>(config["rl"], "state_pattern")[0]);
  calculation_future_ = calculation_promise_.get_future();
  calculation_promise_.set_value(torch::zeros(3, utils::getTensorOptions()));
  if (train) {
    training_service_client = boost::make_shared<ros::ServiceClient>(node_handle.serviceClient<control_force_provider_msgs::UpdateNetwork>("update_network"));
  }
  episode_context_.generateEpisode();
  setGoal(episode_context_.getStart());
  initializing_episode = true;
}

torch::Tensor ReinforcementLearningAgent::getAction() {
  if (train) {
    if (training_service_client->exists()) {
      torch::Tensor state_tensor = state_provider->createState();
      auto state_tensor_accessor = state_tensor.accessor<double, 1>();
      std::vector<double> state_vector;
      for (size_t i = 0; i < state_provider->getStateDim(); i++) state_vector.push_back(state_tensor_accessor[i]);
      control_force_provider_msgs::UpdateNetwork srv;
      srv.request.state = state_vector;
      auto accessor = env.ee_position.accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.robot_position[i] = accessor[i];
      accessor = env.ee_velocity.accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.robot_velocity[i] = accessor[i];
      accessor = env.rcm.accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.robot_rcm[i] = accessor[i];
      for (auto& ob_position : env.ob_positions) {
        accessor = ob_position.accessor<double, 1>();
        for (size_t i = 0; i < ob_position.size(0); i++) srv.request.obstacles_positions.push_back(accessor[i]);
      }
      for (auto& ob_velocity : env.ob_velocities) {
        accessor = ob_velocity.accessor<double, 1>();
        for (size_t i = 0; i < ob_velocity.size(0); i++) srv.request.obstacles_velocities.push_back(accessor[i]);
      }
      for (auto& ob_rcm : env.ob_rcms) {
        accessor = ob_rcm.accessor<double, 1>();
        for (size_t i = 0; i < ob_rcm.size(0); i++) srv.request.obstacles_rcms.push_back(accessor[i]);
      }
      for (auto& point : env.points_on_l1_) {
        accessor = point.accessor<double, 1>();
        for (size_t i = 0; i < point.size(0); i++) srv.request.points_on_l1.push_back(accessor[i]);
      }
      for (auto& point : env.points_on_l2_) {
        accessor = point.accessor<double, 1>();
        for (size_t i = 0; i < point.size(0); i++) srv.request.points_on_l2.push_back(accessor[i]);
      }
      accessor = env.goal.accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.goal[i] = accessor[i];
      auto workspace_bb_origin_acc = env.workspace_bb_origin.accessor<double, 1>();
      auto workspace_bb_dims_acc = env.workspace_bb_dims.accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.workspace_bb_origin[i] = workspace_bb_origin_acc[i];
      for (size_t i = 0; i < 3; i++) srv.request.workspace_bb_dims[i] = workspace_bb_dims_acc[i];
      srv.request.elapsed_time = env.elapsed_time.toDouble();
      if (!training_service_client->call(srv)) ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Failed to call training service.");
      return utils::createTensor(std::vector<double>(std::begin(srv.response.action), std::end(srv.response.action)));
    } else
      return torch::zeros(3, utils::getTensorOptions());
  } else {
    // TODO: extrapolate current state during inference
    // torch::Tensor state = getActionInference(...);
    return torch::zeros(3, utils::getTensorOptions());
  }
}

void ReinforcementLearningAgent::calculationRunnable() { calculation_promise_.set_value(getAction()); }

void ReinforcementLearningAgent::getForceImpl(torch::Tensor& force) {
  if (rcm_origin_ && !env.rcm.equal(torch::zeros(3, utils::getTensorOptions()))) {
    env.setOffset(env.rcm);
    force = current_force_;
    return;
  }
  double now = Time::now();
  if (now - last_calculation_ > interval_duration_) {
    if (train) {
      torch::Tensor goal_vector = env.goal - env.ee_position;
      if (utils::norm(goal_vector).item().toDouble() < goal_reached_threshold_distance_) {
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
    if (!torch::isfinite(current_force_).any().item().toBool() || torch::isnan(current_force_).any().item().toBool()) {
      ROS_WARN_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "The force vector contains Infs or NaNs.");
      current_force_ = torch::nan_to_num(current_force_, 0, 0, 0);
    }
  }
  force = current_force_;
}

DeepQNetworkAgent::DeepQNetworkAgent(const YAML::Node& config, ros::NodeHandle& node_handle) : ReinforcementLearningAgent(config, node_handle) {}

torch::Tensor DeepQNetworkAgent::getActionInference(torch::Tensor& state) {}

MonteCarloAgent::MonteCarloAgent(const YAML::Node& config, ros::NodeHandle& node_handle) : ReinforcementLearningAgent(config, node_handle) {}

torch::Tensor MonteCarloAgent::getActionInference(torch::Tensor& state) { return torch::Tensor(); }
}  // namespace control_force_provider::backend