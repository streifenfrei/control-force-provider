#include "control_force_provider/control_force_calculator.h"

#include <boost/algorithm/clamp.hpp>
#include <boost/chrono.hpp>

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
  Vector4d obstacle_position;
  obstacle->getPosition(obstacle_position);
  Vector3d obstacle_position3d = obstacle_position.head(3);
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
  const Vector3d& a2 = obstacle->getRCM();
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

StateProvider::StateProvider(int obstacle_history_length) : obstacle_history_length_(obstacle_history_length), state_dim_(9 + 3 * obstacle_history_length) {}

void StateProvider::updateObstacleHistory(const Eigen::Vector3d& obstacle_position) {
  do obstacle_history_.push_back(obstacle_position);
  while (obstacle_history_.size() <= obstacle_history_length_);
  obstacle_history_.pop_front();
}

PyObject* StateProvider::createPythonState(const Eigen::Vector3d& ee_position, /*const Eigen::Vector3d& ee_velocity,*/ const Eigen::Vector3d& robot_rcm,
                                           const Eigen::Vector3d& obstacle_position, const Eigen::Vector3d& obstacle_rcm) {
  updateObstacleHistory(obstacle_position);
  PyObject* py_state_ = PyTuple_New(state_dim_);
  unsigned int index = 0;
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(ee_position[0]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(ee_position[1]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(ee_position[2]));
  // PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(ee_velocity[0]));
  // PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(ee_velocity[1]));
  // PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(ee_velocity[2]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(robot_rcm[0]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(robot_rcm[1]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(robot_rcm[2]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(obstacle_rcm[0]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(obstacle_rcm[1]));
  PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(obstacle_rcm[2]));
  for (auto& obstacle : obstacle_history_) {
    PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(obstacle[0]));
    PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(obstacle[1]));
    PyTuple_SetItem(py_state_, index++, PyFloat_FromDouble(obstacle[2]));
  }
  return py_state_;
}

ReinforcementLearningAgent::ReinforcementLearningAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config)
    : ControlForceCalculator(obstacle),
      interval_duration_(ros::Duration(utils::getConfigValue<double>(config, "interval_duration")[0] * 10e-4)),
      train(utils::getConfigValue<bool>(config, "train")[0]),
      output_dir(utils::getConfigValue<std::string>(config, "output_directory")[0]),
      state_provider(utils::getConfigValue<int>(config, "obstacle_history_length")[0]),
      last_calculation_(ros::Time::now()) {
  calculation_future_ = calculation_promise_.get_future();
  calculation_promise_.set_value(Vector4d(0, 0, 0, 0));
  networks_module = loadPythonModule("cfp_networks");
  settings_dict = PyDict_New();
  PyDict_SetItemString(settings_dict, "state_dim", PyLong_FromLong(1));
  PyDict_SetItemString(settings_dict, "action_dim", PyLong_FromLong(1));
  PyDict_SetItemString(settings_dict, "discount_factor", PyFloat_FromDouble(utils::getConfigValue<double>(config, "discount_factor")[0]));
  PyDict_SetItemString(settings_dict, "batch_size", PyLong_FromLong(utils::getConfigValue<int>(config, "batch_size")[0]));
  PyDict_SetItemString(settings_dict, "updates_per_step", PyLong_FromLong(utils::getConfigValue<int>(config, "updates_per_step")[0]));
  PyDict_SetItemString(settings_dict, "max_force", PyFloat_FromDouble(utils::getConfigValue<double>(config, "max_force")[0]));
  PyDict_SetItemString(settings_dict, "output_dir", PyUnicode_FromString(output_dir.c_str()));
}

void ReinforcementLearningAgent::calculationRunnable() { calculation_promise_.set_value(getAction()); }

void ReinforcementLearningAgent::getForceImpl(Vector4d& force) {
  ros::Time now = ros::Time::now();
  if (now - last_calculation_ > interval_duration_) {
    if (train) {
      // TODO pause the simulation here
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
  }
  force = current_force_;
}

DeepQNetworkAgent::DeepQNetworkAgent(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config) : ReinforcementLearningAgent(obstacle, config) {
  ryml::NodeRef dqn_config = utils::getConfigValue<ryml::NodeRef>(config, "dqn")[0];
  if (train) {
    PyDict_SetItemString(settings_dict, "layer_size", PyLong_FromLong(utils::getConfigValue<int>(dqn_config, "layer_size")[0]));
    PyDict_SetItemString(settings_dict, "replay_buffer_size", PyLong_FromLong(utils::getConfigValue<int>(dqn_config, "replay_buffer_size")[0]));
    PyDict_SetItemString(settings_dict, "target_network_update_rate", PyLong_FromLong(utils::getConfigValue<int>(dqn_config, "target_network_update_rate")[0]));
    training_context_ = networks_module.callFunction("DQNContext", nullptr, settings_dict);
  }
}

Vector4d DeepQNetworkAgent::getAction() {
  if (train) {
    Vector4d obstacle_position;
    obstacle->getPosition(obstacle_position);
    PythonObject py_action = training_context_.callFunction(
        "update", Py_BuildValue("(O)", state_provider.createPythonState(ee_position.head(3), rcm, obstacle_position.head(3), obstacle->getRCM())));
    return {PyFloat_AsDouble(PyTuple_GetItem(py_action, 0)), PyFloat_AsDouble(PyTuple_GetItem(py_action, 1)), PyFloat_AsDouble(PyTuple_GetItem(py_action, 2)),
            0};
  } else {
    return {};
  }
}
}  // namespace control_force_provider::backend