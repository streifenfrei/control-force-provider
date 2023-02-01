import os
import torch
from torch.utils.tensorboard import SummaryWriter
from .basics import *
from .monte_carlo import *


class HierarchicalRLContext(RLContext):
    class HighLevelReward:
        reward = None

        def __call__(self, *args, **kwargs):
            zeros = torch.zeros_like(self.reward)
            return self.reward, zeros, zeros, zeros

    class StateProvider:
        def __init__(self, mapping):
            self.mapping = mapping

        def __call__(self, state_dict):
            index, length = self.mapping["gol"]
            state_dict["state"][:, index:index + length] = state_dict["goal"]

    def __init__(self, level_num, algorithm, distance_factor, **kwargs):
        super().__init__(**kwargs)
        assert level_num > 1
        from . import context_mapping
        kwargs["discount_factor"] = (-1 / distance_factor) + 1
        self.agents = [context_mapping[algorithm](**kwargs, log=False)]
        self.reached_goal_threshold_distance = self.agents[0].reward_function.dg
        self.goals = [torch.full([self.robot_batch, 3], torch.nan) for _ in range(level_num)]
        self.rewards = [torch.zeros([self.robot_batch, 1]) for _ in range(level_num)]
        self.last_terminal = torch.zeros([self.robot_batch, 1], dtype=torch.bool)
        self.state_provider = []
        kwargs["max_force"] = float(kwargs["max_force"])
        kwargs["max_force"] *= self.interval_duration
        kwargs["interval_duration"] = 1
        for i in range(1, level_num):
            kwargs["max_force"] *= distance_factor
            self.agents.append(context_mapping[algorithm](**kwargs, log=False))
        for i, agent in enumerate(self.agents):
            agent.summary_writer = SummaryWriter(os.path.join(self.output_dir, "logs", f"level_{i}"))
            self.state_provider.append(self.StateProvider(self.state_augmenter.mapping))
            if i != 0:
                agent.reward_function = self.HighLevelReward()

    def _get_state_dict(self):
        state_dict = {}
        for i, agent in enumerate(self.agents):
            state_dict[f"level_{i}"] = agent._get_state_dict()
        return state_dict

    def _load_impl(self, state_dict):
        for i, agent in enumerate(self.agents):
            agent._load_impl(state_dict[f"level_{i}"])

    @staticmethod
    def _copy_state_dict(state_dict):
        state_dict_copy = {}
        for key, value in state_dict.items():
            state_dict_copy[key] = value.clone()
        return state_dict_copy

    def _update_impl(self, state_dict, reward):
        current_state_dict = state_dict.copy()
        reward = reward.view(-1, 1)
        next_position = None
        last_mask = None
        for i in reversed(range(1, len(self.agents))):
            reached_step = torch.ones([self.robot_batch, 1], dtype=torch.bool) if self.goals[i - 1].isnan().any() \
                else torch.linalg.norm(state_dict["robot_position"] - self.goals[i - 1], dim=-1, keepdims=True) <= self.reached_goal_threshold_distance
            self.rewards[i] += reward
            self.agents[i].reward_function.reward = self.rewards[i].clone()
            if i != len(self.agents) - 1:
                # current_state_dict["is_timeout"] = torch.zeros_like(state_dict["is_timeout"])
                # current_state_dict["abort"] = state_dict["is_timeout"]
                current_state_dict["goal"] = self.goals[i]
            self.state_provider[i](current_state_dict)
            # save old state
            old_agent_state = {}
            for key in ["last_state_dict", "action", "action_index", "real_action_index", "exploration_probs"]:
                if hasattr(self.agents[i], key):
                    old_agent_state[key] = getattr(self.agents[i], key)
            # set masked states to nan
            mask = reached_step.logical_not().squeeze()
            current_state_dict_copy = self._copy_state_dict(current_state_dict)
            current_state_dict_copy["state"][mask, :] = torch.nan
            # update
            self.action = self.agents[i].update(current_state_dict_copy)
            if next_position is not None:
                self.goals[i][last_mask, :] = next_position[last_mask, :]
            # ignore transitions not in the mask
            for old_state_key, old_state_value in old_agent_state.items():
                new_state_value = getattr(self.agents[i], old_state_key)
                if old_state_value is not None and new_state_value is not None:
                    if isinstance(new_state_value, dict):
                        for key, value in old_state_value.items():
                            if value.size(0) == self.robot_batch:
                                new_state_value[key][mask, :] = value[mask, :]
                    else:
                        new_state_value[mask, :] = old_state_value[mask, :]
            if isinstance(self.agents[i], MonteCarloContext):
                for j, episode in enumerate(self.agents[i].episode_buffer):
                    if mask[j] and episode:
                        episode.pop(-1)
            # set next step
            mask = (reached_step.logical_or(self.last_terminal)).squeeze()
            next_position = torch.clamp(state_dict["robot_position"] + self.action, state_dict["workspace_bb_origin"], state_dict["workspace_bb_origin"] + state_dict["workspace_bb_dims"])
            self.rewards[i][mask, :] = 0
            if self.goals[i - 1].isnan().any():
                self.goals[i - 1][mask, :] = next_position[mask, :]
            # setup for next level
            last_mask = mask
            current_state_dict["is_terminal"] = reached_step.logical_or(state_dict["collided"]).logical_or(state_dict["is_timeout"])
            current_state_dict["reached_goal"] = reached_step
        # update level 0
        # current_state_dict["is_timeout"] = torch.zeros_like(state_dict["is_timeout"])
        # current_state_dict["abort"] = state_dict["is_timeout"]
        current_state_dict["goal"] = self.goals[0]
        self.state_provider[0](current_state_dict)
        self.action = self.agents[0].update(self._copy_state_dict(current_state_dict))
        self.goals[0][last_mask, :] = next_position[last_mask, :]
        self.last_terminal = state_dict["is_terminal"]

    def _post_update(self, state_dict):
        for i, agent in enumerate(self.agents):
            for j, (key, value) in enumerate(agent.log_dict.items()):
                self_key = "\n" if j == 0 else ""
                self_key += f"level{i}_{key}"
                if self_key in self.log_dict:
                    self.log_dict[self_key] += value
                else:
                    self.log_dict[self_key] = value
                agent.log_dict[key] = 0
