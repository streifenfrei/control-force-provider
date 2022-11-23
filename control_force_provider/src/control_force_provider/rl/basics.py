import os
import re
import torch
import numpy as np
import rospy
from functools import reduce
from enum import Enum
from abc import ABC, abstractmethod
from concurrent import futures
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation
from collections import namedtuple, defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-9
BB_REPULSION_DISTANCE = 1e-2

Transition = namedtuple('Transition', ('state', 'velocity', 'action', 'next_state', 'reward'))


class StateAugmenter:
    class _StatePartID(str, Enum):
        robot_position = "ree"
        robot_velocity = "rve"
        robot_rotation = "rro"
        robot_rcm = "rpp"
        obstacle_position = "oee"
        obstacle_velocity = "ove"
        obstacle_rotation = "oro"
        obstacle_rcm = "opp"
        goal = "gol"
        time = "tim"

    _pattern_regex = "([a-z]{3})\\((([a-z][0-9]+)*)\\)"
    _arg_regex = "[a-z][0-9]+"

    def __init__(self, state_pattern, num_obstacles, ob_sigma, ob_max_noise):
        self.mapping = {}
        ids = re.findall(self._pattern_regex, state_pattern)
        index = 0
        for id in ids:
            args = id[1]
            id = id[0]
            history_length = 1
            for arg in re.findall(self._arg_regex, args):
                if arg[0] == "h":
                    history_length = int(arg[1:])
            if id == StateAugmenter._StatePartID.time:
                length = 1
            elif id in [StateAugmenter._StatePartID.robot_rotation, StateAugmenter._StatePartID.obstacle_rotation]:
                length = 4
            else:
                length = 3
            length *= num_obstacles if id in [StateAugmenter._StatePartID.obstacle_position,
                                              StateAugmenter._StatePartID.obstacle_velocity,
                                              StateAugmenter._StatePartID.obstacle_rotation,
                                              StateAugmenter._StatePartID.obstacle_rcm] else 1
            length *= history_length
            self.mapping[id] = (index, length)
            index += length
        self.num_obstacles = num_obstacles
        self.rng = np.random.default_rng()
        self.ob_sigma = ob_sigma
        self.ob_max_noise = ob_max_noise

    def __call__(self, state):
        index, length = self.mapping[StateAugmenter._StatePartID.obstacle_position]
        state[:, index:index + length] += np.clip(self.rng.multivariate_normal(np.zeros(length), np.identity(length) * self.ob_sigma, len(state)), -self.ob_max_noise, self.ob_max_noise)
        return state


class RewardFunction:
    def __init__(self, fmax, interval_duration, dc, mc, max_penalty, dg, rg):
        self.fmax = float(fmax)
        self.dc = float(dc)
        self.mc = float(mc)
        self.max_penalty = max_penalty
        self.dg = float(dg)
        self.rg = float(rg)
        self.interval_duration = float(interval_duration)

    def __call__(self, state_dict, last_state_dict):
        goal = np.array(state_dict["goal"])[:3]
        robot_position = np.array(state_dict["robot_position"])[:3]
        last_robot_position = np.array(last_state_dict["robot_position"])[:3]
        distance_vectors = (np.array(state_dict["points_on_l2"][x:x + 3]) - np.array(state_dict["points_on_l1"][x:x + 3]) for x in range(0, len(state_dict["points_on_l1"]), 3))
        distance_to_goal = np.linalg.norm(goal - robot_position)
        motion_reward = (np.linalg.norm(goal - last_robot_position) - distance_to_goal) / (self.fmax * self.interval_duration)
        collision_penalty = reduce(lambda x, y: x + y, ((self.dc / (np.linalg.norm(o) + 1e-10)) ** self.mc for o in distance_vectors))
        collision_penalty = 0 if np.isnan(collision_penalty) or any(np.any(np.isnan(x)) for x in distance_vectors) else - np.minimum(collision_penalty, self.max_penalty)
        goal_reward = 0 if distance_to_goal > self.dg else self.rg
        total_reward = motion_reward + collision_penalty + goal_reward
        return total_reward, motion_reward, collision_penalty, goal_reward


class RLContext(ABC):
    class Accumulator:
        state = 0
        count = 0

        def update_state(self, value):
            self.state += value
            self.count += 1

        def get_value(self):
            return 0 if self.count == 0 else self.state / self.count

        def reset(self):
            self.state = 0
            self.count = 0

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_factor,
                 batch_size,
                 max_force,
                 reward_function,
                 state_augmenter,
                 output_directory,
                 interval_duration,
                 episode_timeout,
                 exploration_angle_sigma,
                 exploration_bb_rep_p,
                 exploration_magnitude_sigma,
                 exploration_decay,
                 exploration_duration,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.max_force = float(max_force)
        self.output_dir = output_directory
        self.save_file = os.path.join(self.output_dir, "save.pt")
        self.summary_writer = SummaryWriter(os.path.join(output_directory, "logs"), max_queue=10000, flush_secs=10)
        self.episode_accumulators = defaultdict(RLContext.Accumulator)
        self.reward_function = reward_function
        self.state_augmenter = state_augmenter
        self.last_state_dict = None
        self.action = None
        self.epoch = 0
        self.episode = 0
        self.episode_start = 0
        self.goal = None
        self.episode_timeout = int(episode_timeout * 1000 / interval_duration)
        self.stop_update = False
        self.thread_executor = futures.ThreadPoolExecutor()
        self.rng = np.random.default_rng()
        self.exploration_angle_sigma = exploration_angle_sigma
        self.exploration_rot_axis = None
        self.exploration_angle = 0
        self.exploration_bb_rep_p = exploration_bb_rep_p
        self.exploration_bb_rep_dims = np.zeros(3)
        self.exploration_magnitude_sigma = exploration_magnitude_sigma
        self.exploration_magnitude = 0
        self.exploration_decay = exploration_decay
        self.exploration_duration = int(exploration_duration * 1000 / interval_duration)
        self.exploration_index = 0
        self.state_batch = None
        self.velocity_batch = None
        self.action_batch = None
        self.reward_batch = None

    def __del__(self):
        self.summary_writer.flush()

    @abstractmethod
    def _get_state_dict(self):
        return

    @abstractmethod
    def _load_impl(self, state_dict):
        return

    @abstractmethod
    def _update_impl(self, state_dict, reward):
        return

    def save(self):
        state_dict = self._get_state_dict()
        state_dict = {**state_dict,
                      "epoch": self.epoch,
                      "episode": self.episode - 1,
                      "exploration_angle_sigma": self.exploration_angle_sigma,
                      "exploration_bb_rep_p": self.exploration_bb_rep_p,
                      "exploration_magnitude_sigma": self.exploration_magnitude_sigma}
        if os.path.exists(self.save_file):
            os.rename(self.save_file, f"{self.save_file}_old")
        torch.save(state_dict, self.save_file)

    def load(self):
        if os.path.exists(self.save_file):
            state_dict = torch.load(self.save_file)
            self.epoch = state_dict["epoch"]
            self.episode = state_dict["episode"]
            self.exploration_angle_sigma = state_dict["exploration_angle_sigma"]
            self.exploration_bb_rep_p = state_dict["exploration_bb_rep_p"]
            self.exploration_magnitude_sigma = state_dict["exploration_magnitude_sigma"]
            self._load_impl(state_dict)

    def update(self, state_dict):
        for key in state_dict:
            state_dict[key] = np.array(state_dict[key])
        goal = state_dict["goal"]
        if (goal != self.goal).any():
            if self.goal is not None:
                for key in self.episode_accumulators:
                    self.summary_writer.add_scalar(key, self.episode_accumulators[key].get_value(), self.episode)
                    self.episode_accumulators[key].reset()
                self.summary_writer.add_scalar("steps_per_episode", self.epoch - self.episode_start, self.episode)
                self.episode += 1
                self.save()
            self.goal = goal
            self.episode_start = self.epoch
            self.last_state_dict = None
        reward = None
        if self.last_state_dict is not None:
            # get reward
            reward, motion_reward, collision_penalty, goal_reward = self.reward_function(state_dict, self.last_state_dict)
            self.summary_writer.add_scalar("reward/per_epoch/total", reward, self.epoch)
            self.episode_accumulators["reward/per_episode/total"].update_state(reward)
            self.summary_writer.add_scalar("reward/per_epoch/motion", motion_reward, self.epoch)
            self.episode_accumulators["reward/per_episode/motion"].update_state(motion_reward)
            self.summary_writer.add_scalar("reward/per_epoch/collision", collision_penalty, self.epoch)
            self.episode_accumulators["reward/per_episode/collision"].update_state(collision_penalty)
            self.summary_writer.add_scalar("reward/per_epoch/goal/", goal_reward, self.epoch)
            self.episode_accumulators["reward/per_episode/goal"].update_state(goal_reward)
        self._update_impl(state_dict, reward)
        self.last_state_dict = state_dict
        if np.any(np.isnan(self.action)):
            rospy.loginfo("RL update yielded NaNs.")
            return self.action
        # check for timeout
        if 0 < self.episode_timeout < self.epoch - self.episode_start:
            self.action = state_dict["goal"][:3] - state_dict["robot_position"][:3]
            self.action = self.action / np.linalg.norm(self.action) * self.max_force
            return self.action
        # exploration
        if self.exploration_index == 0 or self.exploration_rot_axis is None:
            self.exploration_rot_axis = self.rng.multivariate_normal(np.zeros(3), np.identity(3))
            self.exploration_angle = np.deg2rad(self.rng.normal(scale=self.exploration_angle_sigma))
            self.exploration_magnitude = self.rng.normal(scale=self.exploration_magnitude_sigma)
            # set repulsion vector
            self.exploration_bb_rep_dims = np.zeros(3)
            if self.rng.uniform(0., 1.) < self.exploration_bb_rep_p:
                bb_origin = state_dict["workspace_bb_origin"]
                bb_end = bb_origin + state_dict["workspace_bb_dims"]
                ee_position = state_dict["robot_position"][:3]
                for i in range(3):
                    if ee_position[i] - BB_REPULSION_DISTANCE < bb_origin[i]:
                        self.exploration_bb_rep_dims[i] = self.max_force
                    elif ee_position[i] + BB_REPULSION_DISTANCE > bb_end[i]:
                        self.exploration_bb_rep_dims[i] = -self.max_force
        self.exploration_index = (self.exploration_index + 1) % self.exploration_duration
        # change dimensions for which we repulse from the bb wall
        for i in range(3):
            if self.exploration_bb_rep_dims[i] != 0:
                self.action[i] = self.exploration_bb_rep_dims[i]
        self.exploration_bb_rep_p *= self.exploration_decay
        # rotate the action vector a bit
        self.exploration_rot_axis[2] = - (self.exploration_rot_axis[0] * self.action[0] + self.exploration_rot_axis[1] * self.action[1]) / self.action[2]  # make it perpendicular to the action vector
        self.exploration_rot_axis /= np.linalg.norm(self.exploration_rot_axis)
        self.exploration_rot_axis *= self.exploration_angle
        self.exploration_angle_sigma *= self.exploration_decay
        self.action = Rotation.from_rotvec(self.exploration_rot_axis).apply(self.action)
        if self.exploration_angle < 0:
            self.exploration_rot_axis *= -1
        # change magnitude
        magnitude = np.linalg.norm(self.action)
        clipped_magnitude = np.clip(magnitude + self.exploration_magnitude, 0., self.max_force)
        self.summary_writer.add_scalar("magnitude", clipped_magnitude, self.epoch)
        self.action = self.action / magnitude * clipped_magnitude
        self.exploration_magnitude_sigma *= self.exploration_decay
        self.summary_writer.add_scalar("exploration/angle_sigma", self.exploration_angle_sigma, self.epoch)
        self.summary_writer.add_scalar("exploration/magnitude_sigma", self.exploration_magnitude_sigma, self.epoch)
        self.epoch += 1
        return self.action
