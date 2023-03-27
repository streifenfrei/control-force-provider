import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .basics import *
from .dqn import DQN

MonteCarloUpdateTuple = namedtuple('MonteCarloUpdateTuple', ('state', 'action', 'return_', 'weight'))


class MonteCarloContext(DiscreteRLContext):
    def __init__(self, layer_size, soft_is, soft_is_decay, **kwargs):
        super().__init__(**kwargs)
        self.dqn = DQN(self.state_dim, len(self.action_space), layer_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.soft_is = soft_is
        self.soft_is_decay = soft_is_decay
        self.real_action_index = None
        self.buffer_size = 0
        self.ts_model = os.path.join(self.output_dir, "mc_ts.pt")
        self.loss_accumulator = RLContext.Accumulator()
        self.log_dict["loss"] = 0
        self.state_stack = []
        self.actions_stack = []
        self.ee_position_stack = []
        self.weights_stack = []
        self.rewards_stack = []
        self.is_terminal_stack = []
        self.her_terminal_stack = []
        self.ignore_stack = []
        self.abort_stack = []
        self.episode_lengths = torch.zeros([self.robot_batch, 1], device=DEVICE)

    def _get_state_dict_(self):
        return {"model_state_dict": self.dqn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "soft_is": self.soft_is}
        # torch.jit.script(self.dqn).save(self.ts_model)

    def _load_impl_(self, state_dict):
        self.dqn.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.soft_is = state_dict["soft_is"]

    def _update_impl(self, state_dict, reward):
        if self.train:
            if len(self.last_state_dict):
                self.state_stack.append(state_dict["state"])
                self.actions_stack.append(self.action_index)
                was_exploring = self.real_action_index != self.action_index
                weights = torch.where(was_exploring, 0, 1 / self.exploration_probs.gather(-1, self.action_index))
                self.weights_stack.append(weights)
                self.rewards_stack.append(reward)
                self.episode_lengths += 1
                if self.soft_is < EPSILON:
                    self.episode_lengths = torch.where(was_exploring, 0, self.episode_lengths)
                # hindsight experience replay
                self.ee_position_stack.append(state_dict["robot_position"])
                self.her_terminal_stack.append(state_dict["is_timeout"])
                self.episode_lengths *= torch.where(state_dict["is_timeout"], 2, 1)
                # update other buffers
                self.buffer_size += torch.masked_select(self.episode_lengths, state_dict["is_terminal"]).sum()
                self.episode_lengths = torch.where(state_dict["is_terminal"], 0, self.episode_lengths)
                self.is_terminal_stack.append(state_dict["is_terminal"])
                self.ignore_stack.append(torch.zeros_like(was_exploring))
                self.abort_stack.append(state_dict.get("abort", torch.zeros_like(was_exploring)))
            # we're ready for an update
            if self.buffer_size >= self.batch_size:
                time_steps = len(self.actions_stack)
                states = torch.stack(self.state_stack)
                actions = torch.stack(self.actions_stack)
                her_goals = torch.stack(self.ee_position_stack)
                weights = torch.stack(self.weights_stack)
                rewards = torch.stack(self.rewards_stack)
                is_terminal = torch.stack(self.is_terminal_stack)
                her_terminal = torch.stack(self.her_terminal_stack)
                no_ignore = torch.stack(self.ignore_stack).logical_not()
                no_abort = torch.stack(self.abort_stack).logical_not()
                accumulated_weights = torch.ones_like(weights)
                returns = torch.zeros_like(rewards)
                her_returns = torch.zeros_like(rewards)
                is_valid = torch.zeros_like(is_terminal)
                her_is_valid = torch.zeros_like(is_terminal)
                for i in reversed(range(time_steps)):
                    i_plus_one = min(i + 1, time_steps - 1)
                    accumulated_weights[i, :, :] = torch.where(no_ignore[i, :, :],
                                                               torch.where(is_terminal[i, :, :], weights[i, :, :], accumulated_weights[i_plus_one, :, :] * weights[i, :, :]),
                                                               accumulated_weights[i, :, :])
                    returns[i, :, :] = torch.where(no_ignore[i, :, :],
                                                   torch.where(is_terminal[i, :, :], rewards[i, :, :], self.discount_factor * returns[i_plus_one, :, :] + rewards[i, :, :]),
                                                   returns[i, :, :])
                    her_returns[i, :, :] = torch.where(no_ignore[i, :, :],
                                                       torch.where(her_terminal[i, :, :], self.her_reward, self.discount_factor * her_returns[i_plus_one, :, :] + rewards[i, :, :]),
                                                       her_returns[i, :, :])
                    is_valid[i, :, :] = is_valid[i_plus_one, :, :].logical_or(is_terminal[i, :, :]).logical_and(no_abort[i, :, :])
                    her_is_valid[i, :, :] = (her_is_valid[i_plus_one, :, :].logical_and(is_terminal[i, :, :].logical_not())).logical_or(her_terminal[i, :, :])
                    her_goals[i, :, :] = torch.where(her_terminal[i, :, :], her_goals[i, :, :], her_goals[i_plus_one, :, :])
                is_valid = is_valid.logical_and(rewards.isnan().logical_not()).logical_and(no_ignore)
                if self.soft_is < EPSILON:
                    is_valid = is_valid.logical_and(weights > 0)
                her_is_valid = her_is_valid.logical_and(is_valid)
                her_states = states.clone()
                noise = self.her_noise_dist.sample([her_goals.size(1)]).to(DEVICE)
                noise_magnitudes = torch.linalg.vector_norm(noise, dim=-1, keepdims=True)
                noise /= noise_magnitudes
                noise *= torch.minimum(noise_magnitudes, torch.tensor(self.goal_reached_threshold_distance))
                her_states[:, :, self.goal_state_index:self.goal_state_index + 3] = her_goals + noise.unsqueeze(0)
                # stack HER batch on top
                states = torch.cat([states, her_states])
                actions = torch.cat([actions, actions])
                returns = torch.cat([returns, her_returns])
                is_valid = torch.cat([is_valid, her_is_valid])
                accumulated_weights = torch.cat([accumulated_weights, accumulated_weights])
                batch_size = is_valid.sum()
                # create batch
                state_batch = torch.masked_select(states, is_valid).reshape([batch_size, states.size(-1)]).contiguous()
                action_batch = torch.masked_select(actions, is_valid).reshape([batch_size, 1]).contiguous()
                return_batch = torch.masked_select(returns, is_valid).reshape([batch_size, 1]).contiguous()
                weight_batch = (1 - self.soft_is) * torch.masked_select(accumulated_weights, is_valid).reshape([batch_size, 1]).contiguous() + self.soft_is
                # update
                self.optimizer.zero_grad()
                q = self.dqn(state_batch).gather(1, action_batch)
                loss = weight_batch * nn.MSELoss(reduction="none")(q, return_batch)
                loss = loss.mean()
                if not loss.isnan().any():
                    loss.backward()
                    clip_grad_norm_(self.dqn.parameters(), 100)
                    self.optimizer.step()
                    # logging etc.
                    self.soft_is *= self.soft_is_decay
                    self.loss_accumulator.update_state(loss.detach().mean().cpu())
                    self.log_dict["loss"] += self.loss_accumulator.get_value().item()
                    self.loss_accumulator.reset()
                else:
                    self.warn(f"NaNs in MC loss. Epoch {self.epoch}. \n"
                              f"State batch has NaNs: {state_batch.isnan().any()}\n"
                              f"Action batch has NaNs: {action_batch.isnan().any()}\n"
                              f"Return batch has NaNs: {return_batch.isnan().any()}\n"
                              f"Weight batch has NaNs: {weight_batch.isnan().any()}")
                self.buffer_size = 0
                self.state_stack = []
                self.actions_stack = []
                self.ee_position_stack = []
                self.weights_stack = []
                self.rewards_stack = []
                self.is_terminal_stack = []
                self.her_terminal_stack = []
                self.ignore_stack = []
                self.abort_stack = []
                self.episode_lengths = torch.zeros([self.robot_batch, 1], device=DEVICE)
                if self.epoch % self.log_interval == 0:
                    self.summary_writer.add_scalar("soft_is", self.soft_is, self.epoch)
        else:
            self.buffer_size = 0
            self.state_stack = []
            self.actions_stack = []
            self.ee_position_stack = []
            self.weights_stack = []
            self.rewards_stack = []
            self.is_terminal_stack = []
            self.her_terminal_stack = []
            self.ignore_stack = []
            self.abort_stack = []
            self.episode_lengths = torch.zeros([self.robot_batch, 1], device=DEVICE)
        self.dqn.eval()
        with torch.no_grad():
            self.action_index = self.dqn(state_dict["state"]).max(-1)[1].unsqueeze(-1)
            self.real_action_index = self.action_index
        self.dqn.train()
