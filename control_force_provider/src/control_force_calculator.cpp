#include "control_force_provider/control_force_calculator.h"

#include <boost/algorithm/clamp.hpp>
#include <boost/array.hpp>
#include <boost/chrono.hpp>
#include <string>

#include "control_force_provider/utils.h"

using namespace Eigen;
namespace control_force_provider::backend {

PotentialFieldMethod::PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config)
    : ControlForceCalculator(obstacle),
      attraction_strength_(utils::getConfigValue<double>(config, "attraction_strength")[0]),
      attraction_distance_(utils::getConfigValue<double>(config, "attraction_distance")[0]),
      repulsion_strength_(utils::getConfigValue<double>(config, "repulsion_strength")[0]),
      repulsion_distance_(utils::getConfigValue<double>(config, "repulsion_distance")[0]),
      z_translation_strength_(utils::getConfigValue<double>(config, "z_translation_strength")[0]),
      min_rcm_distance_(utils::getConfigValue<double>(config, "min_rcm_distance")[0]) {}

void PotentialFieldMethod::getForceImpl(Vector4d& force) {
  Vector3d obstacle_position3d = ob_position.head(3);
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
  //  for the vector between "robot" and "obstacle" we take the shortest line between both tools:
  //  tool1:                                                    l1 = a1 + t*b1
  //  tool2:                                                    l2 = a2 + s*b2
  const Vector3d& a1 = rcm;
  Vector3d b1 = ee_position3d - a1;
  const Vector3d& a2 = ob_rcm;
  Vector3d b2 = obstacle_position3d - a2;
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

  Vector3d repulsive_vector = {0, 0, 0};
  if (t > 0) {  // both points lie past the RCMs (=inside the abdomen). If not we don't need to calculate the vector anyway
    //  use the first equation to get s:
    //                s*e1 = t*e3 - b1•a'                                     |: e1
    //                ———————————————————
    //                s = (t*e3 - b1•a') / e1
    double s = (t * e3 - b1.dot(a_diff)) / e1;
    t = boost::algorithm::clamp(t, 0, 1);
    s = boost::algorithm::clamp(s, 0, 1);
    point_on_l1_ = (a1 + t * b1);
    point_on_l2_ = (a2 + s * b2);
    Vector3d l2_to_l1 = point_on_l1_ - point_on_l2_;
    double l2_to_l1_distance = l2_to_l1.norm();
    // continue with PFM
    if (l2_to_l1_distance < repulsion_distance_) {
      double l1_length = b1.norm();
      repulsive_vector = (repulsion_strength_ / l2_to_l1_distance - repulsion_strength_ / repulsion_distance_) / (l2_to_l1_distance * l2_to_l1_distance) *
                         l2_to_l1.normalized() * t * l1_length;
      // avoid positive z-translation
      Vector3d l1_new = ee_position3d + repulsive_vector - a1;
      double l1_new_length = l1_new.norm();
      if (l1_length - l1_new_length < 0) {
        l1_new *= l1_length / l1_new_length;
        repulsive_vector = a1 + l1_new - ee_position3d;
      }
      // add negative z-translation
      repulsive_vector -= (z_translation_strength_ / l2_to_l1_distance - z_translation_strength_ / repulsion_distance_) /
                          (l2_to_l1_distance * l2_to_l1_distance) * (b1 / l1_length);
      if ((ee_position3d + repulsive_vector - a1).norm() < min_rcm_distance_)  // prevent pushing the end effector too close to the RCM
        repulsive_vector = {0, 0, 0};
    }
  }
  Vector3d force3d = attractive_vector + repulsive_vector;
  force = {force3d[0], force3d[1], force3d[2], 0};
}

StateProvider::StateProvider(const Vector4d& ee_position, const Vector3d& robot_rcm, const Vector4d& obstacle_position, const Vector3d& obstacle_rcm,
                             const Vector4d& goal, const std::string& state_pattern) {
  for (auto& str : utils::regex_findall(pattern_regex, state_pattern)) {
    std::string id = str.substr(0, 3);
    std::string args = str.substr(4, str.length() - 5);
    if (id == "ree") {
      state_populators_.emplace_back(ee_position.data(), 3);
    } else if (id == "rpp") {
      state_populators_.emplace_back(robot_rcm.data(), 3);
    } else if (id == "oee") {
      state_populators_.emplace_back(obstacle_position.data(), 3);
    } else if (id == "opp") {
      state_populators_.emplace_back(obstacle_rcm.data(), 3);
    } else if (id == "gol") {
      state_populators_.emplace_back(goal.data(), 3);
    } else {
      ROS_WARN_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Unknown id in state pattern: " << id);
      continue;
    }
    // parse arguments
    for (auto& arg : utils::regex_findall(arg_regex, args)) {
      std::string arg_id = arg.substr(0, 1);
      unsigned int value = std::stoi(arg.substr(1, arg.length() - 1));
      if (arg_id == "h")
        state_populators_.back().history_length_ = value;
      else if (arg_id == "s")
        state_populators_.back().history_stride_ = value;
      else
        ROS_WARN_STREAM_NAMED("control_force_provider/control_force_calculator/rl", "Unknown argument " << arg << " for id " << id << " in state pattern");
    }
  }
  for (auto& pop : state_populators_) state_dim_ += pop.getDim();
}

void StateProvider::StatePopulator::populate(torch::Tensor& state, int& index) {
  if (stride_index_ == 0) {
    VectorXd eigen_vector(length_);
    for (size_t i = 0; i < length_; i++) eigen_vector[i] = vector_[i];
    do history_.push_back(eigen_vector);
    while (history_.size() <= history_length_);
    history_.pop_front();
    stride_index_ = history_stride_;
  } else
    stride_index_ -= 1;
  auto state_accessor = state.accessor<double, 1>();
  for (auto& vec : history_)
    for (size_t i = 0; i < length_; i++) state_accessor[index++] = vec[i];
}

torch::Tensor StateProvider::createState() {
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor state = torch::empty(state_dim_, options);
  int index = 0;
  for (auto& pop : state_populators_) pop.populate(state, index);
  return state;
}

ReinforcementLearningAgent::ReinforcementLearningAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config, ros::NodeHandle& node_handle)
    : ControlForceCalculator(obstacle),
      interval_duration_(ros::Duration(utils::getConfigValue<double>(config, "interval_duration")[0] * 10e-4)),
      train(utils::getConfigValue<bool>(config, "train")[0]),
      output_dir(utils::getConfigValue<std::string>(config, "output_directory")[0]),
      last_calculation_(ros::Time::now()) {
  state_provider =
      boost::make_shared<StateProvider>(ee_position, rcm, ob_position, ob_rcm, goal, utils::getConfigValue<std::string>(config, "state_pattern")[0]);
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

DeepQNetworkAgent::DeepQNetworkAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config, ros::NodeHandle& node_handle)
    : ReinforcementLearningAgent(obstacle, config, node_handle) {}

torch::Tensor DeepQNetworkAgent::getActionInference(torch::Tensor& state) {}
}  // namespace control_force_provider::backend