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

Environment::Environment(const YAML::Node& config, int batch_size, torch::DeviceType device)
    : ee_position_(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU))),
      ee_rotation_(torch::zeros({batch_size, 4}, utils::getTensorOptions(torch::kCPU))),
      ee_velocity_(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU))),
      rcm_(torch::zeros(3, utils::getTensorOptions(device_))),
      goal_(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU))),
      start_time_(torch::zeros({batch_size, 1}, utils::getTensorOptions(torch::kCPU))),
      last_update_(0),
      elapsed_time_(torch::zeros({batch_size, 1}, utils::getTensorOptions(torch::kCPU))),
      workspace_bb_origin_(utils::createTensor(utils::getConfigValue<double>(config, "workspace_bb"), 0, 3, device)),
      workspace_bb_dims_(utils::createTensor(utils::getConfigValue<double>(config, "workspace_bb"), 3, 6, device)),
      max_force_(utils::getConfigValue<double>(config, "max_force")[0]),
      offset_(torch::zeros({1, 3}, utils::getTensorOptions(device_))),
      device_(device) {
  std::string data_path;
  obstacles_ = Obstacle::createFromConfig(config, data_path, batch_size, device_);
  std::vector<boost::shared_ptr<FramesObstacle>> frames_obstacles;
  int reference_obstacle = -1;
  for (auto& obstacle : obstacles_) {
    ob_rcms_.push_back(obstacle->getRCM());
    ob_positions_.push_back(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU)));
    ob_directions_.push_back(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU)));
    ob_rotations_.push_back(torch::zeros({batch_size, 4}, utils::getTensorOptions(torch::kCPU)));
    ob_velocities_.push_back(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU)));
    points_on_l1_.push_back(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU)));
    points_on_l2_.push_back(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU)));
    collision_distances_.push_back(torch::zeros({batch_size, 1}, utils::getTensorOptions(torch::kCPU)));
    boost::shared_ptr<FramesObstacle> frames_obstacle = boost::dynamic_pointer_cast<FramesObstacle>(obstacle);
    if (frames_obstacle) {
      if (frames_obstacle->getRCM().equal(torch::zeros(3, utils::getTensorOptions(torch::kCPU)))) {
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

void Environment::update(const torch::Tensor& ee_position) {
  torch::Tensor ee_position_dev = ee_position.to(device_) - offset_;
  double now = Time::now();
  double time_since_last_update = now - last_update_;
  if (time_since_last_update > 0) {
    boost::lock_guard<boost::shared_mutex> lock(ee_vel_mtx);
    ee_velocity_ = ((ee_position_dev - ee_position_.to(device_)) / time_since_last_update).cpu();
  }
  last_update_ = now;
  {
    boost::lock_guard<boost::shared_mutex> lock(ee_pos_mtx);
    ee_position_ = ee_position_dev.cpu();
  }
  {
    boost::lock_guard<boost::shared_mutex> lock(ee_rot_mtx);
    ee_rotation_ = utils::zRotation(rcm_ - offset_, ee_position_dev);
  }
  elapsed_time_ = (now - start_time_.to(device_)).cpu();
  std::vector<torch::Tensor> ob_rcms_dev;
  std::vector<torch::Tensor> ob_positions_dev;
  for (size_t i = 0; i < obstacles_.size(); i++) {
    torch::Tensor ob_position_dev = obstacles_[i]->getPosition().to(device_);
    ob_position_dev -= offset_;
    ob_positions_dev.push_back(ob_position_dev);
    {
      boost::lock_guard<boost::shared_mutex> lock(ob_vel_mtx);
      ob_velocities_[i] = (ob_position_dev - ob_positions_[i].to(device_)).cpu();
    }
    {
      boost::lock_guard<boost::shared_mutex> lock(ob_pos_mtx);
      ob_positions_[i] = ob_position_dev.cpu();
    }
    torch::Tensor ob_rcm_dev = (obstacles_[i]->getRCM().to(device_) - offset_);
    ob_rcms_dev.push_back(ob_rcm_dev);
    {
      boost::lock_guard<boost::shared_mutex> lock(ob_rcm_mtx);
      ob_rcms_[i] = ob_rcm_dev.cpu();
    }
    {
      boost::lock_guard<boost::shared_mutex> lock(ob_dir_mtx);
      ob_directions_[i] = (ob_rcm_dev - ob_position_dev).cpu();
    }
    {
      boost::lock_guard<boost::shared_mutex> lock(ob_rot_mtx);
      ob_rotations_[i] = utils::zRotation(ob_rcm_dev, ob_position_dev).cpu();
    }
  }
  torch::Tensor a1 = rcm_.to(device_);
  torch::Tensor b1 = ee_position_dev - a1;
  for (size_t i = 0; i < obstacles_.size(); i++) {
    torch::Tensor t, s = torch::empty(ee_position_.size(0), utils::getTensorOptions(device_));
    const torch::Tensor& a2 = ob_rcms_dev[i];
    torch::Tensor b2 = ob_positions_dev[i] - a2;
    utils::shortestLine(a1, b1, a2, b2, t, s);
    t = torch::clamp(t, 0, 1);
    s = torch::clamp(s, 0, 1);
    torch::Tensor point_on_l1_dev = a1 + t * b1;
    {
      boost::lock_guard<boost::shared_mutex> lock(l1_mtx);
      points_on_l1_[i] = point_on_l1_dev.cpu();
    }
    torch::Tensor point_on_l2_dev = a2 + s * b2;
    {
      boost::lock_guard<boost::shared_mutex> lock(l2_mtx);
      points_on_l2_[i] = point_on_l2_dev;
    }
    {
      boost::lock_guard<boost::shared_mutex> lock(cd_mtx);
      collision_distances_[i] = utils::norm((point_on_l1_dev - point_on_l2_dev)).cpu();
    }
  }
}

void Environment::clipForce(torch::Tensor& force) {
  torch::Tensor force_dev = force.to(device_);
  torch::Tensor magnitude = utils::norm(force_dev);
  torch::Tensor mask = magnitude > max_force_;
  force_dev = torch::where(mask, force / magnitude * max_force_, force_dev);
  torch::Tensor ee_position_dev = ee_position_.to(device_);
  torch::Tensor next_pos = ee_position_dev + force_dev;
  torch::Tensor distance_to_wall = torch::full_like(force_dev, INFINITY);
  torch::Tensor next_distance_to_wall = torch::full_like(force_dev, INFINITY);
  distance_to_wall = torch::min(distance_to_wall, ee_position_dev - workspace_bb_origin_);
  distance_to_wall = torch::min(distance_to_wall, workspace_bb_origin_ + workspace_bb_dims_ - ee_position_dev);
  distance_to_wall = std::get<0>(torch::min(distance_to_wall, -1, true));
  next_distance_to_wall = torch::min(next_distance_to_wall, next_pos - workspace_bb_origin_);
  next_distance_to_wall = torch::min(next_distance_to_wall, workspace_bb_origin_ + workspace_bb_dims_ - next_pos);
  next_distance_to_wall = std::get<0>(torch::min(next_distance_to_wall, -1, true));
  next_pos = torch::clamp(next_pos, workspace_bb_origin_, workspace_bb_origin_ + workspace_bb_dims_);
  mask = next_distance_to_wall < distance_to_wall;
  if (torch::any(mask).item().toBool()) {
    force_dev = torch::where(mask, next_pos - ee_position_dev, force_dev);
    mask = torch::logical_and(mask, distance_to_wall > 0);
    if (torch::any(mask).item().toBool()) {
      torch::Tensor max_magnitude = distance_to_wall * workspace_bb_stopping_strength;
      magnitude = utils::norm(force_dev);
      mask = torch::logical_and(mask, magnitude > max_magnitude);
      if (torch::any(mask).item().toBool()) force_dev = torch::where(mask, force_dev / magnitude * max_magnitude, force_dev);
    }
  }
  force = force_dev.to(force.device());
}

const torch::Tensor& Environment::getEePosition() const { return ee_position_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getEePositionLock() { return boost::shared_lock_guard<boost::shared_mutex>(ee_pos_mtx); }
const torch::Tensor& Environment::getEeRotation() const { return ee_rotation_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getEeRotationLock() { return boost::shared_lock_guard<boost::shared_mutex>(ee_rot_mtx); }
const torch::Tensor& Environment::getEeVelocity() const { return ee_velocity_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getEeVelocityLock() { return boost::shared_lock_guard<boost::shared_mutex>(ee_vel_mtx); }
const torch::Tensor& Environment::getRCM() const { return rcm_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getRCMLock() { return boost::shared_lock_guard<boost::shared_mutex>(rcm_mtx); }
void Environment::setRCM(const torch::Tensor& rcm) { rcm_ = rcm.to(device_) - offset_; }
const torch::Tensor& Environment::getGoal() const { return goal_; }
void Environment::setGoal(const torch::Tensor& goal) { goal_ = (goal.to(device_) - offset_).cpu(); }
boost::shared_lock_guard<boost::shared_mutex> Environment::getGoalLock() { return boost::shared_lock_guard<boost::shared_mutex>(goal_mtx); }
const torch::Tensor& Environment::getWorkspaceBbOrigin() const { return workspace_bb_origin_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getWorkspaceBbOriginLock() { return boost::shared_lock_guard<boost::shared_mutex>(bb_or_mtx); }
const std::vector<torch::Tensor>& Environment::getObPositions() const { return ob_positions_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getObDirectionsLock() { return boost::shared_lock_guard<boost::shared_mutex>(ob_dir_mtx); }
const std::vector<torch::Tensor>& Environment::getObDirections() const { return ob_directions_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getObPositionsLock() { return boost::shared_lock_guard<boost::shared_mutex>(ob_pos_mtx); }
const std::vector<torch::Tensor>& Environment::getObRotations() const { return ob_rotations_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getObRotationsLock() { return boost::shared_lock_guard<boost::shared_mutex>(ob_rot_mtx); }
const std::vector<torch::Tensor>& Environment::getObVelocities() const { return ob_velocities_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getObVelocitiesLock() { return boost::shared_lock_guard<boost::shared_mutex>(goal_mtx); }
const std::vector<torch::Tensor>& Environment::getObRCMs() const { return ob_rcms_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getObRCMsLock() { return boost::shared_lock_guard<boost::shared_mutex>(ob_rcm_mtx); }
const std::vector<torch::Tensor>& Environment::getPointsOnL1() const { return points_on_l1_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getPointsOnL1Lock() { return boost::shared_lock_guard<boost::shared_mutex>(l1_mtx); }
const std::vector<torch::Tensor>& Environment::getPointsOnL2() const { return points_on_l2_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getPointsOnL2Lock() { return boost::shared_lock_guard<boost::shared_mutex>(goal_mtx); }
const std::vector<torch::Tensor>& Environment::getCollisionDistances() const { return collision_distances_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getCollisionDistancesLock() { return boost::shared_lock_guard<boost::shared_mutex>(cd_mtx); }
const torch::Tensor& Environment::getOffset() const { return offset_; }
boost::shared_lock_guard<boost::shared_mutex> Environment::getOffsetLock() { return boost::shared_lock_guard<boost::shared_mutex>(offset_mtx); }
void Environment::setOffset(const torch::Tensor& offset) {
  // TODO fix it
  torch::Tensor offset_dev = offset.to(device_);
  torch::Tensor translation = offset_ - offset_dev;
  workspace_bb_origin_ += translation;
  rcm_ += translation;
  goal_ = (goal_.to(device_) + translation).cpu();
  offset_ = offset_dev;
}
const torch::Tensor& Environment::getWorkspaceBbDims() const { return workspace_bb_dims_; }
double Environment::getMaxForce() const { return max_force_.toFloat(); }
const torch::Tensor& Environment::getElapsedTime() const { return elapsed_time_; }
const torch::Tensor& Environment::getStartTime() const { return start_time_; }
void Environment::setStartTime(const torch::Tensor& startTime) {
  torch::Tensor start_time_dev = startTime.to(device_);
  start_time_ = startTime.cpu();
  elapsed_time_ = (Time::now() - start_time_dev).cpu();
}
const std::vector<boost::shared_ptr<Obstacle>>& Environment::getObstacles() const { return obstacles_; }
const boost::shared_ptr<ObstacleLoader>& Environment::getObstacleLoader() const { return obstacle_loader_; }

ControlForceCalculator::ControlForceCalculator(const YAML::Node& config) : env(boost::make_shared<Environment>(config)), device_(torch::hasCUDA() ? torch::kCUDA : torch::kCPU) {}
ControlForceCalculator::ControlForceCalculator(boost::shared_ptr<Environment> env_) : env(env_), device_(torch::hasCUDA() ? torch::kCUDA : torch::kCPU) {}

void ControlForceCalculator::getForce(torch::Tensor& force, const torch::Tensor& ee_position_) {
  if (!goal_available_) setGoal(ee_position_.to(device_) - env->getOffset());
  if (rcm_available_) {
    env->update(ee_position_);
    getForceImpl(force);
    env->clipForce(force);
  } else {
    force = torch::zeros_like(force).to(device_);
  }
}

PotentialFieldMethod::PotentialFieldMethod(const YAML::Node& config)
    : ControlForceCalculator(config),
      repulsion_only_(false),
      attraction_strength_(utils::getConfigValue<double>(config["pfm"], "attraction_strength")[0]),
      attraction_distance_(utils::getConfigValue<double>(config["pfm"], "attraction_distance")[0]),
      repulsion_strength_(utils::getConfigValue<double>(config["pfm"], "repulsion_strength")[0]),
      repulsion_distance_(utils::getConfigValue<double>(config["pfm"], "repulsion_distance")[0]),
      z_translation_strength_(utils::getConfigValue<double>(config["pfm"], "z_translation_strength")[0]),
      min_rcm_distance_(utils::getConfigValue<double>(config["pfm"], "min_rcm_distance")[0]) {}
PotentialFieldMethod::PotentialFieldMethod(const YAML::Node& config, boost::shared_ptr<Environment> env_)
    : ControlForceCalculator(env_),
      repulsion_only_(false),
      attraction_strength_(utils::getConfigValue<double>(config["pfm"], "attraction_strength")[0]),
      attraction_distance_(utils::getConfigValue<double>(config["pfm"], "attraction_distance")[0]),
      repulsion_strength_(utils::getConfigValue<double>(config["pfm"], "repulsion_strength")[0]),
      repulsion_distance_(utils::getConfigValue<double>(config["pfm"], "repulsion_distance")[0]),
      z_translation_strength_(utils::getConfigValue<double>(config["pfm"], "z_translation_strength")[0]),
      min_rcm_distance_(utils::getConfigValue<double>(config["pfm"], "min_rcm_distance")[0]) {}

void PotentialFieldMethod::getForceImpl(torch::Tensor& force) {
  // attractive vector
  torch::Tensor ee_position_dev = env->getEePosition().to(device_);
  torch::Tensor attractive_vector = env->getGoal().to(device_) - ee_position_dev;
  torch::Tensor ee_to_goal_distance = utils::norm(attractive_vector);
  torch::Tensor is_linear_attraction = ee_to_goal_distance > attraction_distance_;
  torch::Tensor smoothing_factor = torch::where(is_linear_attraction, 0.1, 1);
  attractive_vector = torch::where(is_linear_attraction, attractive_vector / ee_to_goal_distance, attractive_vector);
  attractive_vector *= attraction_strength_ * smoothing_factor;
  // repulsive vector
  torch::Tensor repulsive_vector = torch::zeros_like(force);
  torch::Tensor rcm_dev = env->getRCM().to(device_);
  const torch::Tensor& a1 = rcm_dev;
  torch::Tensor b1 = ee_position_dev - a1;
  torch::Tensor l1_length = utils::norm(b1);
  torch::Tensor min_l2_to_l1_distance = torch::full_like(l1_length, repulsion_distance_);
  for (size_t i = 0; i < env->getObPositions().size(); i++) {
    torch::Tensor point_on_l1_dev = env->getPointsOnL1()[i].to(device_);
    torch::Tensor t = utils::norm(point_on_l1_dev - rcm_dev) / l1_length;
    torch::Tensor l2_to_l1 = point_on_l1_dev - env->getPointsOnL2()[i].to(device_);
    torch::Tensor l2_to_l1_distance = utils::norm(l2_to_l1);
    torch::Tensor is_repulsing = (l2_to_l1_distance < repulsion_distance_).expand({-1, 3});
    repulsive_vector += torch::where(is_repulsing,
                                     (repulsion_strength_ / l2_to_l1_distance - repulsion_strength_ / repulsion_distance_) /
                                         (l2_to_l1_distance * l2_to_l1_distance) * l2_to_l1 / utils::norm(l2_to_l1) * t * l1_length,
                                     0);
    min_l2_to_l1_distance = torch::minimum(min_l2_to_l1_distance, l2_to_l1_distance);
  }
  torch::Tensor is_repulsing = min_l2_to_l1_distance < repulsion_distance_;
  // avoid positive z-translation
  torch::Tensor l1_new = ee_position_dev + repulsive_vector - a1;
  torch::Tensor l1_new_length = utils::norm(l1_new);
  torch::Tensor positive_z_translation = l1_length - l1_new_length < 0;
  l1_new *= torch::where(positive_z_translation, l1_length / l1_new_length, 1);
  repulsive_vector = torch::where(is_repulsing.logical_and(positive_z_translation), a1 + l1_new - ee_position_dev, repulsive_vector);

  // add negative z-translation
  repulsive_vector -= torch::where(is_repulsing,
                                   (z_translation_strength_ / min_l2_to_l1_distance - z_translation_strength_ / repulsion_distance_) /
                                       (min_l2_to_l1_distance * min_l2_to_l1_distance) * (b1 / l1_length),
                                   0);
  // prevent pushing the end effector too close to the RCM
  repulsive_vector = torch::where((utils::norm((ee_position_dev + repulsive_vector - a1)) < min_rcm_distance_).expand({-1, 3}), 0, repulsive_vector);
  force = repulsive_vector;
  if (!repulsion_only_) force += attractive_vector;
}

StateProvider::StatePopulator StateProvider::createPopulatorFromString(const Environment& env, const std::string& str, torch::DeviceType device) {
  std::string id = str.substr(0, 3);
  std::string args = str.substr(4, str.length() - 5);
  StatePopulator populator(device);
  populator.length_ = 3;
  if (id == "ree") {
    populator.tensors_.push_back(&env.getEePosition());
  } else if (id == "rve") {
    populator.tensors_.push_back(&env.getEeVelocity());
  } else if (id == "rro") {
    populator.length_ = 4;
    populator.tensors_.push_back(&env.getEeRotation());
  } else if (id == "rpp") {
    populator.tensors_.push_back(&env.getRCM());
  } else if (id == "oee") {
    for (auto& pos : env.getObPositions()) populator.tensors_.push_back(&pos);
  } else if (id == "ove") {
    for (auto& vel : env.getObVelocities()) populator.tensors_.push_back(&vel);
  } else if (id == "odi") {
    for (auto& dir : env.getObDirections()) populator.tensors_.push_back(&dir);
  } else if (id == "oro") {
    populator.length_ = 4;
    for (auto& rot : env.getObRotations()) populator.tensors_.push_back(&rot);
  } else if (id == "opp") {
    for (auto& rcm : env.getObRCMs()) populator.tensors_.push_back(&rcm);
  } else if (id == "gol") {
    populator.tensors_.push_back(&env.getGoal());
  } else if (id == "tim") {
    populator.length_ = 1;
    populator.tensors_.push_back(&env.getElapsedTime());
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

StateProvider::StateProvider(const Environment& env, const std::string& state_pattern, torch::DeviceType device) {
  for (auto& str : utils::regexFindAll(pattern_regex, state_pattern)) {
    state_populators_.push_back(createPopulatorFromString(env, str, device));
    state_dim_ += state_populators_.back().getDim();
  }
}

StateProvider::StatePopulator::StatePopulator(torch::DeviceType device): device_(device) {};

void StateProvider::StatePopulator::populate(torch::Tensor& state) {
  for (size_t i = 0; i < tensors_.size(); i++) {
    const torch::Tensor* tensor = tensors_[i];
    if (histories_.size() <= i) histories_.emplace_back();
    std::deque<torch::Tensor>& history = histories_[i];
    if (stride_index_ == 0) {
      do history.push_back(*tensor);
      while (history.size() <= history_length_);
      history.pop_front();
    }
    for (auto& ten : history) state = state.defined() ? torch::cat({state, ten.to(device_)}, -1) : ten.to(device_);
  }
  stride_index_ = (stride_index_ - 1) % (history_stride_ + 1);
}

torch::Tensor StateProvider::createState() {
  torch::Tensor state;
  for (auto& pop : state_populators_) pop.populate(state);
  return torch::where(state.isnan(), 0, state).cpu();
}

EpisodeContext::EpisodeContext(std::vector<boost::shared_ptr<Obstacle>> obstacles, boost::shared_ptr<ObstacleLoader> obstacle_loader, const YAML::Node& config,
                               unsigned int batch_size, torch::DeviceType device)
    : obstacles_(obstacles),
      obstacle_loader_(obstacle_loader),
      begin_max_offset_(utils::getConfigValue<double>(config, "begin_max_offset")[0]),
      start_(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU))),
      goal_(torch::zeros({batch_size, 3}, utils::getTensorOptions(torch::kCPU))),
      start_bb_origin(utils::createTensor(utils ::getConfigValue<double>(config, "start_bb"), 0, 3, device)),
      start_bb_dims(utils::createTensor(utils::getConfigValue<double>(config, "start_bb"), 3, 6, device)),
      goal_bb_origin(utils::createTensor(utils::getConfigValue<double>(config, "goal_bb"), 0, 3, device)),
      goal_bb_dims(utils::createTensor(utils::getConfigValue<double>(config, "goal_bb"), 3, 6, device)),
      device_(device) {}

void EpisodeContext::generateEpisode(const torch::Tensor& mask) {
  torch::Tensor mask_ = mask.to(device_);
  torch::Tensor start_dev = start_.to(device_);
  torch::Tensor goal_dev = goal_.to(device_);
  start_ = torch::where(mask_, start_bb_origin + start_bb_dims * torch::rand_like(start_dev), start_dev);
  goal_ = torch::where(mask_, goal_bb_origin + goal_bb_dims * torch::rand_like(goal_dev), goal_dev);
  obstacle_loader_->loadNext();
}

void EpisodeContext::generateEpisode() { this->generateEpisode(torch::ones({start_.size(0), 1}, utils::getTensorOptions(device_, torch::kBool))); }

void EpisodeContext::startEpisode(const torch::Tensor& mask) {
  torch::Tensor offset = begin_max_offset_ * torch::rand({start_.size(0), 1}, utils::getTensorOptions(device_));
  torch::Tensor mask_ = mask.to(device_);
  for (auto& obstacle : obstacles_) obstacle->reset(mask_, offset);
}

void EpisodeContext::startEpisode() { this->startEpisode(torch::ones({start_.size(0), 1}, utils::getTensorOptions(device_, torch::kBool))); }

ReinforcementLearningAgent::ReinforcementLearningAgent(const YAML::Node& config, ros::NodeHandle& node_handle)
    : ControlForceCalculator(config),
      interval_duration_(utils::getConfigValue<double>(config["rl"], "interval_duration")[0] * 10e-4),
      goal_reached_threshold_distance_(utils::getConfigValue<double>(config["rl"], "goal_reached_threshold_distance")[0]),
      episode_context_(boost::make_shared<EpisodeContext>(env->getObstacles(), env->getObstacleLoader(), config["rl"])),
      train(utils::getConfigValue<bool>(config["rl"], "train")[0]),
      rcm_origin_(utils::getConfigValue<bool>(config["rl"], "rcm_origin")[0]),
      output_dir(utils::getConfigValue<std::string>(config["rl"], "output_directory")[0]),
      current_force_(torch::zeros(3, utils::getTensorOptions())),
      last_calculation_(Time::now()) {
  if (rcm_origin_) env->setOffset(env->getRCM());
  state_provider = boost::make_shared<StateProvider>(*env, utils::getConfigValue<std::string>(config["rl"], "state_pattern")[0]);
  calculation_future_ = calculation_promise_.get_future();
  calculation_promise_.set_value(torch::zeros(3, utils::getTensorOptions()));
  if (train) {
    training_service_client = boost::make_shared<ros::ServiceClient>(node_handle.serviceClient<control_force_provider_msgs::UpdateNetwork>("update_network"));
  }
  episode_context_->generateEpisode();
  setGoal(episode_context_->getStart());
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
      auto accessor = env->getEePosition().accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.robot_position[i] = accessor[i];
      accessor = env->getEePosition().accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.robot_velocity[i] = accessor[i];
      accessor = env->getRCM().accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.robot_rcm[i] = accessor[i];
      for (auto& ob_position : env->getObPositions()) {
        accessor = ob_position.accessor<double, 1>();
        for (size_t i = 0; i < ob_position.size(0); i++) srv.request.obstacles_positions.push_back(accessor[i]);
      }
      for (auto& ob_velocity : env->getObVelocities()) {
        accessor = ob_velocity.accessor<double, 1>();
        for (size_t i = 0; i < ob_velocity.size(0); i++) srv.request.obstacles_velocities.push_back(accessor[i]);
      }
      for (auto& ob_rcm : env->getObRCMs()) {
        accessor = ob_rcm.accessor<double, 1>();
        for (size_t i = 0; i < ob_rcm.size(0); i++) srv.request.obstacles_rcms.push_back(accessor[i]);
      }
      for (auto& point : env->getPointsOnL1()) {
        accessor = point.accessor<double, 1>();
        for (size_t i = 0; i < point.size(0); i++) srv.request.points_on_l1.push_back(accessor[i]);
      }
      for (auto& point : env->getPointsOnL2()) {
        accessor = point.accessor<double, 1>();
        for (size_t i = 0; i < point.size(0); i++) srv.request.points_on_l2.push_back(accessor[i]);
      }
      accessor = env->getGoal().accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.goal[i] = accessor[i];
      auto workspace_bb_origin_acc = env->getWorkspaceBbOrigin().accessor<double, 1>();
      auto workspace_bb_dims_acc = env->getWorkspaceBbDims().accessor<double, 1>();
      for (size_t i = 0; i < 3; i++) srv.request.workspace_bb_origin[i] = workspace_bb_origin_acc[i];
      for (size_t i = 0; i < 3; i++) srv.request.workspace_bb_dims[i] = workspace_bb_dims_acc[i];
      srv.request.elapsed_time = env->getElapsedTime().item().toFloat();
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
  if (rcm_origin_ && !env->getRCM().equal(torch::zeros(3, utils::getTensorOptions()))) {
    env->setOffset(env->getRCM());
    force = current_force_;
    return;
  }
  double now = Time::now();
  if (now - last_calculation_ > interval_duration_) {
    if (train) {
      torch::Tensor goal_vector = env->getGoal() - env->getEePosition();
      if (utils::norm(goal_vector).item().toFloat() < goal_reached_threshold_distance_) {
        if (initializing_episode) {
          episode_context_->startEpisode();
          setGoal(episode_context_->getGoal());
          initializing_episode = false;
        } else {
          goal_delay_count++;
          if (goal_delay_count >= goal_delay) {
            episode_context_->generateEpisode();
            setGoal(episode_context_->getStart());
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