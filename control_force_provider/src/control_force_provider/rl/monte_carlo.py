import random
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
        self.episode_buffer = [[] for _ in range(self.robot_batch)]  # contains transitions for unfinished episodes
        self.update_buffer = []  # contains finished episodes
        self.update_buffer_size = 0
        self.ts_model = os.path.join(self.output_dir, "mc_ts.pt")
        self.loss_accumulator = RLContext.Accumulator()
        self.log_dict["loss"] = 0

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
        # populate episode_buffer and update_buffer
        if self.last_state_dict is not None:
            last_states = torch.split(self.last_state_dict["state"], 1)
            actions = torch.split(self.action_index, 1)
            was_exploring = self.real_action_index != self.action_index
            weights = torch.where(was_exploring, 0,
                                  torch.where(self.action_index < len(self.action_space.dynamic_space),
                                              1 / (1 - (self.exploration_epsilon * self.exploration_goal_p / len(self.action_space.dynamic_space))),
                                              1 / (1 - (self.exploration_epsilon * (1 - self.exploration_goal_p) / len(self.action_space.static_space))))).split(1)
            rewards = torch.split(reward, 1)
            for i in range(self.robot_batch):
                if self.soft_is < EPSILON and was_exploring[i]:
                    self.episode_buffer[i].clear()
                    continue
                if not rewards[i].isnan():
                    self.episode_buffer[i].append(MonteCarloUpdateTuple(last_states[i], actions[i], rewards[i], weights[i]))
                elif state_dict["is_terminal"][i]:
                    self.episode_buffer[i].reverse()
                    self.update_buffer.append(self.episode_buffer[i])
                    episode_length = len(self.episode_buffer[i])
                    self.update_buffer_size += episode_length
                    self.episode_buffer[i] = []
        # we're ready for an update
        if self.update_buffer_size >= self.batch_size:
            update_transitions = []
            weights = torch.ones([len(self.update_buffer), 1], device=DEVICE)
            returns = torch.zeros([len(self.update_buffer), 1], device=DEVICE)
            step = 0
            # calculate returns and importance sampling weights for all transitions
            while self.update_buffer:
                for i, transitions in enumerate(zip(*self.update_buffer)):
                    if i < step:
                        continue
                    batch = MonteCarloUpdateTuple(*zip(*transitions))
                    returns = self.discount_factor * returns + torch.cat(batch.return_)
                    weights *= torch.cat(batch.weight)
                    update_transitions += [MonteCarloUpdateTuple(state, action, return_, weight)
                                           for state, action, return_, weight in
                                           zip(batch.state, batch.action, returns.split(1), weights.split(1))]
                    step += 1
                keep_indices = [x for x in range(len(self.update_buffer)) if len(self.update_buffer[x]) > step]
                if keep_indices:
                    weights = torch.stack([weights[x, :] for x in keep_indices])
                    returns = torch.stack([returns[x, :] for x in keep_indices])
                self.update_buffer = [self.update_buffer[x] for x in keep_indices]
            # create batch
            update_transitions = random.sample(update_transitions, self.batch_size)
            batch = MonteCarloUpdateTuple(*zip(*update_transitions))
            state_batch = self.state_augmenter(torch.cat(batch.state)).to(DEVICE)
            action_batch = torch.cat(batch.action).to(DEVICE)
            return_batch = torch.cat(batch.return_).to(DEVICE)
            weight_batch = (1 - self.soft_is) * torch.cat(batch.weight).to(DEVICE) + self.soft_is
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
                self.summary_writer.add_scalar("loss/total", self.loss_accumulator.get_value(), self.epoch)
                self.summary_writer.add_scalar("exploration/soft_is", self.soft_is, self.epoch)
                self.loss_accumulator.reset()
            else:
                rospy.logwarn(f"NaNs in MC loss. Epoch {self.epoch}")
            self.episode_buffer = [[] for _ in range(self.robot_batch)]
            self.update_buffer = []
            self.update_buffer_size = 0
        self.dqn.eval()
        with torch.no_grad():
            self.action_index = self.dqn(state_dict["state"]).max(-1)[1].unsqueeze(-1)
            self.real_action_index = self.action_index
        self.dqn.train()
