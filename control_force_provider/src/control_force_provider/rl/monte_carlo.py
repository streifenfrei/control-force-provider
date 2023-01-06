import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .basics import *
from .dqn import DQN


class MonteCarloContext(RLContext):
    def __init__(self, layer_size, **kwargs):
        super().__init__(**kwargs)
        self.dqn = DQN(self.state_dim, self.action_dim, layer_size, self.max_force).to(device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.total_reward = 0
        self.episode_buffer = []
        self.update_buffer = []
        self.ts_model = os.path.join(self.output_dir, "mc_ts.pt")
        self.loss_accumulator = RLContext.Accumulator()

    def _get_state_dict(self):
        return {"model_state_dict": self.dqn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()}
        # torch.jit.script(self.dqn).save(self.ts_model)

    def _load_impl(self, state_dict):
        self.dqn.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def _update_impl(self, state_dict, reward, is_terminal):
        # TODO: use tensors batches
        if reward is not None:
            self.total_reward += reward
        else:
            for transition in self.episode_buffer:
                if len(self.update_buffer) < self.batch_size:
                    self.update_buffer.append(Transition(transition.state, None, transition.action, None, self.total_reward))
                else:
                    batch = Transition(*zip(*self.update_buffer))
                    state_batch = torch.tensor(self.state_augmenter(np.stack(batch.state)), dtype=torch.float32).to(device)
                    action_batch = torch.tensor(np.stack(batch.action), dtype=torch.float32).to(device)
                    reward_batch = torch.tensor(np.stack(batch.reward), dtype=torch.float32).unsqueeze(-1).to(device)
                    mu, q, _ = self.dqn(state_batch, action_batch)
                    loss = nn.MSELoss()(q, reward_batch)
                    self.summary_writer.add_scalar("loss", loss.detach().cpu().numpy(), self.episode - 1)
                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.dqn.parameters(), 1)
                    self.optimizer.step()
                    self.update_buffer = []
                    break
            self.episode_buffer = []
            self.total_reward = 0
        self.dqn.eval()
        with torch.no_grad():
            self.action = self.dqn(torch.tensor(state_dict["state"], dtype=torch.float32).unsqueeze(0).to(device))[0].squeeze(0).cpu().numpy()
        self.dqn.train()
        self.episode_buffer.append(Transition(state_dict["state"], None, self.action, None, None))
