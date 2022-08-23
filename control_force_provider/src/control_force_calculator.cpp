#include "control_force_provider/control_force_calculator.h"

#include <boost/algorithm/clamp.hpp>
#include <boost/array.hpp>
#include <boost/chrono.hpp>
#include <string>
#include <utility>

#include "control_force_provider/utils.h"

using namespace Eigen;
namespace control_force_provider::backend {

ControlForceCalculator::ControlForceCalculator(std::vector<boost::shared_ptr<Obstacle>> obstacles_) : obstacles(std::move(obstacles_)) {
  for (auto& obstacle : obstacles) {
    ob_rcms.push_back(obstacle->getRCM());
    ob_positions.emplace_back();
  }
}

void ControlForceCalculator::getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position_) {
  if (!goal_available_) setGoal(ee_position);
  if (rcm_available_) {
    ee_position = ee_position_;
    for (size_t i = 0; i < obstacles.size(); i++) {
      obstacles[i]->getPosition(ob_positions[i]);
      ob_rcms[i] = obstacles[i]->getRCM();
    }
    getForceImpl(force);
  }
}

PotentialFieldMethod::PotentialFieldMethod(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config)
    : ControlForceCalculator(std::move(obstacles_)),
      attraction_strength_(utils::getConfigValue<double>(config, "attraction_strength")[0]),
      attraction_distance_(utils::getConfigValue<double>(config, "attraction_distance")[0]),
      repulsion_strength_(utils::getConfigValue<double>(config, "repulsion_strength")[0]),
      repulsion_distance_(utils::getConfigValue<double>(config, "repulsion_distance")[0]),
      z_translation_strength_(utils::getConfigValue<double>(config, "z_translation_strength")[0]),
      min_rcm_distance_(utils::getConfigValue<double>(config, "min_rcm_distance")[0]) {
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
    Vector3d ob_position = ob_positions[i].head(3);
    //  for the vector between "robot" and "obstacle" we take the shortest line between both tools:
    //  tool1:                                                    l1 = a1 + t*b1
    //  tool2:                                                    l2 = a2 + s*b2
    const Vector3d& a2 = ob_rcms[i];
    Vector3d b2 = ob_position - a2;
    //  general vector between l1 and l2:                         v = a2 - a1 + sb2 - t*b1 = a' + s*b2 - t*b1
    Vector3d a_diff = a2 - a1;
    //  the shortest line is perpendicular to both tools (v•b1 = v•b2 = 0). We want to solve this LEQ for t and s:
    //                b1•a' + s*b1•b2 - t*b1•b1 = 0
    //                b2•a' + s*b2•b2 - t*b2•b1 = 0
    //  substitute e1 = b1•b2, e2 = b2•b2 and e3 = b1•b1
    double e1 = b1.dot(b2);
    double e2 = b2.dot(b2);
    double e3 = b1.dot(b1);
    //                b1•a' + s*e1 - t*e3 = 0                                 |* e2
    //                b2•a' + s*e2 - t*e1 = 0                                 |* e1
    //                ————————————————————————————
    //                b1•a'*e2 + s*e1*e2 - t*e2*e3 = 0                        -
    //                b2•a'*e1 + s*e1*e2 - t*e1*e1 = 0
    //                ————————————————————————————————
    //                b1•a'*e2 - t*e2*e3 - b2•a'*e1 + t*e1*e1 = 0             |+ b2•a'*e1, - b1•a'*e2
    //                ———————————————————————————————————————————
    //                t*(e1*e1 - e2*e3) = a'•(b2*e1 - b1*e2)                  |: (e1*e1 - b1•b1*e2)
    //                ——————————————————————————————————————
    //                t = a'•(b2*e1 - b1*e2) / (e1*e1 - e2*e3)
    double t = a_diff.dot(b2 * e1 - b1 * e2) / (e1 * e1 - e2 * e3);

    if (t > 0) {  // both points lie past the RCMs (=inside the abdomen). If not we don't need to calculate the vector anyway
      //  use the first equation to get s:
      //                s*e1 = t*e3 - b1•a'                                     |: e1
      //                ———————————————————
      //                s = (t*e3 - b1•a') / e1
      double s = (t * e3 - b1.dot(a_diff)) / e1;
      t = boost::algorithm::clamp(t, 0, 1);
      s = boost::algorithm::clamp(s, 0, 1);
      points_on_l1_[i] = (a1 + t * b1);
      points_on_l2_[i] = (a2 + s * b2);

      Vector3d l2_to_l1 = points_on_l1_[i] - points_on_l2_[i];
      double l2_to_l1_distance = l2_to_l1.norm();
      // continue with PFM
      if (l2_to_l1_distance < repulsion_distance_) {
        repulsive_vector += (repulsion_strength_ / l2_to_l1_distance - repulsion_strength_ / repulsion_distance_) / (l2_to_l1_distance * l2_to_l1_distance) *
                            l2_to_l1.normalized() * t * l1_length;
        if (l2_to_l1_distance < min_l2_to_l1_distance) min_l2_to_l1_distance = l2_to_l1_distance;
      }
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
  } else if (id == "rpp") {
    populator.vectors_.push_back(cfc.rcm.data());
  } else if (id == "oee") {
    for (auto& pos : cfc.ob_positions) populator.vectors_.push_back(pos.data());
  } else if (id == "opp") {
    for (auto& rcm : cfc.ob_rcms) populator.vectors_.push_back(rcm.data());
  } else if (id == "gol") {
    populator.vectors_.push_back(cfc.goal.data());
  } else {
    ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Unknown id in state pattern: " << id);
  }
  // parse arguments
  for (auto& arg : utils::regex_findall(arg_regex, args)) {
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
  for (auto& str : utils::regex_findall(pattern_regex, state_pattern)) {
    state_populators_.push_back(createPopulatorFromString(cfc, str));
    state_dim_ += state_populators_.back().getDim();
  }
}

void StateProvider::StatePopulator::populate(torch::Tensor& state, int& index) {
  auto state_accessor = state.accessor<double, 1>();
  for (size_t i = 0; i < histories_.size(); i++) {
    const double* vector = vectors_[i];
    if (histories_.size() <= i) histories_.emplace_back();
    std::deque<Eigen::VectorXd>& history = histories_[i];
    if (stride_index_ == 0) {
      VectorXd eigen_vector(length_);
      for (size_t j = 0; j < length_; j++) eigen_vector[j] = vector[j];
      do history.push_back(eigen_vector);
      while (history.size() <= history_length_);
      history.pop_front();
    }
    for (auto& vec : history)
      for (size_t j = 0; j < length_; j++) state_accessor[index++] = vec[j];
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

ReinforcementLearningAgent::ReinforcementLearningAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config,
                                                       ros::NodeHandle& node_handle)
    : ControlForceCalculator(std::move(obstacles_)),
      interval_duration_(ros::Duration(utils::getConfigValue<double>(config, "interval_duration")[0] * 10e-4)),
      train(utils::getConfigValue<bool>(config, "train")[0]),
      output_dir(utils::getConfigValue<std::string>(config, "output_directory")[0]),
      last_calculation_(ros::Time::now()) {
  state_provider = boost::make_shared<StateProvider>(*this, utils::getConfigValue<std::string>(config, "state_pattern")[0]);
  calculation_future_ = calculation_promise_.get_future();
  calculation_promise_.set_value(Vector4d::Zero());
  if (train) {
    training_service_client = boost::make_shared<ros::ServiceClient>(node_handle.serviceClient<control_force_provider_msgs::UpdateNetwork>("update_network"));
  }
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
      for (size_t i = 0; i < 4; i++) srv.request.goal[i] = goal[i];
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
  ros::Time now = ros::Time::now();
  if (now - last_calculation_ > interval_duration_) {
    if (train) {
      current_force_ = getAction();
    } else {
      if (calculation_future_.wait_for(boost::chrono::seconds(0)) == boost::future_status::ready) {
        last_calculation_ = now;
        current_force_ = calculation_future_.get();
        calculation_promise_ = {};
        calculation_future_ = calculation_promise_.get_future();
        boost::thread calculation_thread{&ReinforcementLearningAgent::calculationRunnable, this};
      } else {
        ROS_ERROR_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "The RL agent exceeded the time limit!");
      }
    }
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

DeepQNetworkAgent::DeepQNetworkAgent(std::vector<boost::shared_ptr<Obstacle>> obstacles_, const YAML::Node& config, ros::NodeHandle& node_handle)
    : ReinforcementLearningAgent(std::move(obstacles_), config, node_handle) {}

torch::Tensor DeepQNetworkAgent::getActionInference(torch::Tensor& state) {}
}  // namespace control_force_provider::backend