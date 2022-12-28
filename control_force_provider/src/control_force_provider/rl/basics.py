import os
import re
import torch
import numpy as np
import rospy
from enum import Enum
from abc import ABC, abstractmethod
from concurrent import futures
from torch.utils.tensorboard import SummaryWriter
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
        obstacle_direction = "odi"
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
                                              StateAugmenter._StatePartID.obstacle_direction,
                                              StateAugmenter._StatePartID.obstacle_rcm] else 1
            length *= history_length
            self.mapping[id] = (index, length)
            index += length
        self.num_obstacles = num_obstacles
        self.ob_sigma = ob_sigma
        self.ob_max_noise = ob_max_noise
        self.noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * self.ob_sigma)

    def __call__(self, state):
        index, length = self.mapping[StateAugmenter._StatePartID.obstacle_position]
        stack = []
        for i in range(0, length, 3):
            stack.append(self.noise_dist.sample([state.size(0)]).to(state.device))
        state[:, index:index + length] += torch.clip(torch.cat(stack, 1), -self.ob_max_noise, self.ob_max_noise)
        return state


class RewardFunction:
    def __init__(self, fmax, interval_duration, dc, mc, max_penalty, dg, rg):
        self.fmax = torch.tensor(float(fmax), device=DEVICE)
        self.dc = torch.tensor(float(dc), device=DEVICE)
        self.mc = torch.tensor(float(mc), device=DEVICE)
        self.max_penalty = torch.tensor(float(max_penalty), device=DEVICE)
        self.dg = torch.tensor(float(dg), device=DEVICE)
        self.rg = torch.tensor(float(rg), device=DEVICE)
        self.interval_duration = float(interval_duration)

    def __call__(self, state_dict, last_state_dict, mask):
        goal = state_dict["goal"]
        if last_state_dict is None:
            out = torch.full([goal.size(0)], torch.nan)
            return out, out, out, out
        robot_position = state_dict["robot_position"]
        last_robot_position = last_state_dict["robot_position"]
        distance_vectors = (state_dict["points_on_l2"][:, x:x + 3] - state_dict["points_on_l1"][:, x:x + 3] for x in range(0, state_dict["points_on_l1"].size(-1), 3))
        distance_to_goal = torch.linalg.norm(goal - robot_position)
        motion_reward = torch.where(mask, (torch.linalg.norm(goal - last_robot_position) - distance_to_goal) / (self.fmax * self.interval_duration), torch.nan)
        collision_penalty = torch.where(mask, 0, torch.nan)
        for distance_vector in distance_vectors:
            distance_vector = torch.where(distance_vector.isnan(), 0, distance_vector)
            collision_penalty += (self.dc / (torch.linalg.norm(distance_vector) + EPSILON)) ** self.mc
        collision_penalty = torch.minimum(collision_penalty, self.max_penalty)
        goal_reward = torch.where(distance_to_goal > self.dg, torch.zeros_like(collision_penalty), self.rg)
        total_reward = motion_reward + collision_penalty + goal_reward
        return total_reward, motion_reward, collision_penalty, goal_reward


class RLContext(ABC):
    class Accumulator:
        def __init__(self, batch_size=1):
            self.state = torch.zeros(batch_size, device=DEVICE)
            self.count = torch.zeros(batch_size, device=DEVICE)
            self.default_mask = torch.ones_like(self.state, device=DEVICE, dtype=torch.bool)

        def update_state(self, value, mask=None):
            if mask is None:
                mask = self.default_mask
            mask = mask.to(DEVICE)
            value = value.to(DEVICE)
            self.state = torch.where(mask, self.state + value, self.state)
            self.count = torch.where(mask, self.count + 1, self.count)

        def get_value(self, mask=None):
            if mask is None:
                mask = self.default_mask
            value = torch.where(torch.logical_and(mask, self.count != 0), self.state / self.count, 0)
            return torch.masked_select(value, mask).cpu().numpy()

        def reset(self, mask=None):
            if mask is None:
                mask = self.default_mask
            self.state = torch.where(mask, 0, self.state)
            self.count = torch.where(mask, 0, self.count)

    class AccumulatorFactory:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def __call__(self):
            return RLContext.Accumulator(self.batch_size)

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_factor,
                 batch_size,
                 robot_batch,
                 max_force,
                 reward_function,
                 state_augmenter,
                 output_directory,
                 save_rate,
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
        self.robot_batch = robot_batch
        self.max_force = float(max_force)
        self.output_dir = output_directory
        self.save_rate = save_rate
        self.summary_writer = SummaryWriter(os.path.join(output_directory, "logs"), max_queue=10000, flush_secs=10)
        self.log_interval = 10000
        self.log_dict = {"reward": 0}
        self.episode_accumulators = defaultdict(RLContext.AccumulatorFactory(robot_batch))
        self.reward_function = reward_function
        self.state_augmenter = state_augmenter
        self.last_state_dict = None
        self.action = None
        self.epoch = 0
        self.episode = torch.zeros((robot_batch, 1), device=DEVICE)
        self.episode_start = torch.zeros((robot_batch, 1), device=DEVICE)
        self.total_episode_count = 0
        self.goal = None
        self.interval_duration = interval_duration
        self.episode_timeout = int(episode_timeout * 1000 / interval_duration)
        self.stop_update = False
        self.thread_executor = futures.ThreadPoolExecutor()
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
        self.exploration_rot_axis_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3))
        self.exploration_bb_rep_dist = torch.distributions.uniform.Uniform(0, 1)
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
                      "episode": self.total_episode_count - 1,
                      "exploration_angle_sigma": self.exploration_angle_sigma,
                      "exploration_bb_rep_p": self.exploration_bb_rep_p,
                      "exploration_magnitude_sigma": self.exploration_magnitude_sigma}
        torch.save(state_dict, os.path.join(self.output_dir, f"{self.epoch}.pt"))

    def load(self):
        save_file = None
        max_epoch = 0
        for file in os.listdir(self.output_dir):
            if os.path.isfile(file) and file.endswith(".pt"):
                epoch = int(file[:-3])
                if epoch >= max_epoch:
                    save_file = file
                    max_epoch = epoch
        if save_file is not None:
            state_dict = torch.load(save_file)
            self.epoch = state_dict["epoch"]
            self.episode_start = torch.full_like(self.episode_start, self.epoch)
            self.total_episode_count = state_dict["episode"]
            self.exploration_angle_sigma = state_dict["exploration_angle_sigma"]
            self.exploration_bb_rep_p = state_dict["exploration_bb_rep_p"]
            self.exploration_magnitude_sigma = state_dict["exploration_magnitude_sigma"]
            self._load_impl(state_dict)

    def update(self, state_dict):
        state_dict["state"] = self.state_augmenter(state_dict["state"])
        goal = state_dict["goal"]
        if self.goal is None:
            self.goal = goal
        # check for finished episodes
        finished = goal.not_equal(self.goal).any(-1, True)
        not_finished = finished.logical_not()
        if finished.any():
            for key in self.episode_accumulators:
                for i, value in enumerate(self.episode_accumulators[key].get_value(finished)):
                    self.summary_writer.add_scalar(key, value, self.total_episode_count + i)
                self.episode_accumulators[key].reset(finished)
            steps_per_episode = torch.masked_select(self.epoch - self.episode_start, finished).cpu().numpy()
            for value in steps_per_episode:
                self.summary_writer.add_scalar("steps_per_episode", value, self.total_episode_count)
                self.total_episode_count += 1
            self.goal = goal
            self.episode_start = torch.where(finished, self.epoch, self.episode_start)
        # get reward
        reward, motion_reward, collision_penalty, goal_reward = self.reward_function(state_dict, self.last_state_dict, not_finished)
        total_reward_mean = torch.masked_select(reward, torch.logical_not(reward.isnan())).mean().cpu()
        self.log_dict["reward"] += total_reward_mean
        self.summary_writer.add_scalar("reward/per_epoch/total", total_reward_mean, self.epoch)
        self.episode_accumulators["reward/per_episode/total"].update_state(reward, not_finished)
        self.summary_writer.add_scalar("reward/per_epoch/motion", torch.masked_select(motion_reward, torch.logical_not(motion_reward.isnan())).mean(), self.epoch)
        self.episode_accumulators["reward/per_episode/motion"].update_state(motion_reward, not_finished)
        self.summary_writer.add_scalar("reward/per_epoch/collision", torch.masked_select(collision_penalty, torch.logical_not(collision_penalty.isnan())).mean(), self.epoch)
        self.episode_accumulators["reward/per_episode/collision"].update_state(collision_penalty, not_finished)
        self.summary_writer.add_scalar("reward/per_epoch/goal/", torch.masked_select(goal_reward, torch.logical_not(goal_reward.isnan())).mean(), self.epoch)
        self.episode_accumulators["reward/per_episode/goal"].update_state(goal_reward, not_finished)
        # do the update
        self._update_impl(state_dict, reward)
        self.last_state_dict = state_dict
        nans = torch.isnan(self.action)
        if torch.any(nans):
            #rospy.logwarn(f"NaNs in action tensor. Epoch {self.epoch}")
            return torch.where(nans, 0, self.action)
        # check for timeout
        timeout = None
        if self.episode_timeout > 0:
            timeout = self.epoch - self.episode_start > self.episode_timeout
            if timeout.any():
                self.action = torch.where(timeout, state_dict["goal"] - state_dict["robot_position"], self.action)
                distance_to_goal = torch.linalg.norm(self.action)
                self.action = torch.where(timeout, self.action / distance_to_goal * torch.tensor(self.max_force), self.action)
        explore = torch.ones_like(self.episode_start, dtype=torch.bool) if timeout is None else timeout.logical_not()
        # exploration
        if self.exploration_index == 0 or self.exploration_rot_axis is None:
            self.exploration_rot_axis = self.exploration_rot_axis_dist.sample([self.robot_batch]).to(DEVICE)
            self.exploration_angle = torch.deg2rad(torch.distributions.normal.Normal(loc=0, scale=self.exploration_angle_sigma).sample([self.robot_batch])).unsqueeze(-1).to(DEVICE)
            self.exploration_magnitude = torch.distributions.normal.Normal(loc=0, scale=self.exploration_magnitude_sigma).sample([self.robot_batch]).unsqueeze(-1).to(DEVICE)
            # set repulsion vector
            self.exploration_bb_rep_dims = torch.empty_like(self.action)
            bb_rep = self.exploration_bb_rep_dist.sample(finished.shape).to(DEVICE) < self.exploration_bb_rep_dims
            bb_origin = state_dict["workspace_bb_origin"]
            bb_end = bb_origin + state_dict["workspace_bb_dims"]
            ee_position = state_dict["robot_position"]
            mask = ee_position - BB_REPULSION_DISTANCE < bb_origin
            self.exploration_bb_rep_dims = torch.where(bb_rep.logical_and(mask), self.max_force, 0)
            mask = mask.logical_not().logical_and(ee_position + BB_REPULSION_DISTANCE > bb_end)
            self.exploration_bb_rep_dims = torch.where(bb_rep.logical_and(mask), -self.max_force, self.exploration_bb_rep_dims)
        self.exploration_index = (self.exploration_index + 1) % self.exploration_duration
        # change dimensions for which we repulse from the bb wall
        self.action = torch.where((self.exploration_bb_rep_dims != 0).logical_and(explore), self.exploration_bb_rep_dims, self.action)
        self.exploration_bb_rep_p *= self.exploration_decay
        # rotate the action vector a bit
        self.exploration_rot_axis[:, 2] = - (self.exploration_rot_axis[:, 0] * self.action[:, 0] + self.exploration_rot_axis[:, 1] * self.action[:, 1]) / self.action[:, 2]  # make it perpendicular to the action vector
        self.exploration_rot_axis /= torch.linalg.norm(self.exploration_rot_axis, -1)
        self.exploration_angle_sigma *= self.exploration_decay
        cos_angle = torch.cos(self.exploration_angle)
        self.action = torch.where(explore,
                                  self.action * cos_angle +  # Rodrigues' rotation formula
                                  torch.sin(self.exploration_angle) * torch.linalg.cross(self.action, self.exploration_rot_axis) +
                                  self.exploration_rot_axis * torch.linalg.vecdot(self.exploration_rot_axis, self.action).unsqueeze(-1) * (1 - cos_angle),
                                  self.action)
        self.exploration_rot_axis = torch.where(self.exploration_angle < 0, self.exploration_rot_axis * -1, self.exploration_rot_axis)
        # change magnitude
        magnitude = torch.linalg.norm(self.action)
        clipped_magnitude = torch.clip(magnitude + self.exploration_magnitude, 0., self.max_force)
        self.summary_writer.add_scalar("magnitude", clipped_magnitude.mean(), self.epoch)
        self.action = torch.where(explore, self.action / magnitude * clipped_magnitude, self.action)
        self.exploration_magnitude_sigma *= self.exploration_decay
        self.summary_writer.add_scalar("exploration/angle_sigma", self.exploration_angle_sigma, self.epoch)
        self.summary_writer.add_scalar("exploration/magnitude_sigma", self.exploration_magnitude_sigma, self.epoch)
        # log
        if self.epoch > 0:
            if self.epoch % self.log_interval == 0:
                string = f"Epoch {self.epoch} | "
                for key in self.log_dict:
                    string += f"{key}: {self.log_dict[key] / self.log_interval}\t "
                    self.log_dict[key] = 0
                rospy.loginfo(string)
            if self.epoch % self.save_rate == 0:
                self.save()
        self.epoch += 1
        return self.action
