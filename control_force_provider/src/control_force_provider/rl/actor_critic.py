import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .basics import *
from .dqn import ReplayBuffer

ACTransition = namedtuple('Transition', ('state', 'velocity', 'action', 'next_state', 'reward', 'exploration_prob'))


class Actor(nn.Module):
    def __init__(self, state_dim, out_dim, layer_size):
        super(Actor, self).__init__()
        self.dense1 = nn.Linear(state_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.out = nn.Linear(layer_size, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        x = self.bnorm2(x)
        x = 10 * torch.tanh(self.out(x))
        return self.softmax(x)


class Critic(nn.Module):
    def __init__(self, state_dim, layer_size):
        super(Critic, self).__init__()
        self.dense1 = nn.Linear(state_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.out = nn.Linear(layer_size, 1)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        x = self.bnorm2(x)
        return self.out(x)


class A2CContext(DiscreteRLContext):
    def __init__(self,
                 actor_layer_size,
                 critic_layer_size,
                 replay_buffer_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = Actor(self.state_dim, len(self.action_space), actor_layer_size).to(DEVICE)
        self.critic = Critic(self.state_dim, critic_layer_size).to(DEVICE)
        self.kld = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.replay_buffer = ReplayBuffer(replay_buffer_size, transition=ACTransition)
        self.log_dict["loss/critic"] = 0
        self.log_dict["loss/actor"] = 0
        torch.autograd.set_detect_anomaly(True)

    def _get_state_dict_(self):
        return {"actor_model_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_model_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict()}

    def _load_impl_(self, state_dict):
        self.actor.load_state_dict(state_dict["actor_model_state_dict"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer_state_dict"])
        self.critic.load_state_dict(state_dict["critic_model_state_dict"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer_state_dict"])

    def get_correction_weight(self, p, q):
        m = torch.log((0.5 * (p + q)))
        jsd = 0.5 * (self.kld(m, torch.log(p)) + self.kld(m, torch.log(q)))
        return torch.exp(-jsd)

    def _update_impl(self, state_dict, reward):
        if self.train:
            if self.last_state_dict is not None:
                next_state = torch.where(state_dict["is_terminal"].to(DEVICE).expand(-1, state_dict["state"].size(-1)), torch.nan, state_dict["state"].to(DEVICE))
                self.replay_buffer.push(self.last_state_dict["state"], self.last_state_dict["robot_velocity"], self.action_index, next_state, reward, self.exploration_probs.gather(-1, self.action_index))
                # hindsight experience replay
                her_transition = self.create_her_transitions(self.last_state_dict, self.action_index, state_dict, reward, single_transition=True)
                if her_transition is not None:
                    her_state, her_velocity, her_action, _, _ = her_transition
                    self.actor.eval()
                    with torch.no_grad():
                        her_exploration_probs = self.get_exploration_probs(self.actor(her_state), her_velocity).gather(-1, her_action)
                    self.actor.train()
                    self.replay_buffer.push(*her_transition, her_exploration_probs)

            if len(self.replay_buffer) >= self.batch_size:
                for batch in self.replay_buffer.sample(self.batch_size)():
                    state_batch = batch.state.to(DEVICE)
                    action_batch = batch.action.to(DEVICE)
                    next_state_batch = batch.next_state.to(DEVICE)
                    reward_batch = batch.reward.to(DEVICE)
                    exploration_prob_batch = batch.exploration_prob.to(DEVICE)
                    is_terminal = next_state_batch.isnan().any(-1, keepdims=True).float()
                    next_state_batch = torch.where(next_state_batch.isnan(), 0, next_state_batch)
                    advantage = reward_batch + (1. - is_terminal) * self.discount_factor * self.critic(next_state_batch) - self.critic(state_batch)
                    action_probs = self.actor(state_batch)
                    # off-policy correction
                    action_probs = action_probs.gather(-1, action_batch)
                    weights = self.get_correction_weight(action_probs, exploration_prob_batch).detach()
                    weights_sum = weights.sum()
                    # loss calculation
                    critic_loss = torch.linalg.norm(advantage * weights) / weights_sum
                    actor_loss = torch.log(action_probs) * advantage.detach() * weights
                    actor_loss = -actor_loss.sum() / weights_sum
                    if not critic_loss.isnan().any():
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        clip_grad_norm_(self.critic.parameters(), 1)
                        self.critic_optimizer.step()
                        self.log_dict["loss/critic"] += critic_loss.item()
                    else:
                        rospy.logwarn(f"NaNs in critic loss. Epoch {self.epoch}")
                    if not actor_loss.isnan().any():
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        clip_grad_norm_(self.actor.parameters(), 1)
                        self.actor_optimizer.step()
                        self.log_dict["loss/actor"] += actor_loss.item()
                    else:
                        rospy.logwarn(f"NaNs in critic loss. Epoch {self.epoch}")
        self.actor.eval()
        with torch.no_grad():
            self.action_index = self.actor(state_dict["state"].to(DEVICE))
        self.actor.train()
