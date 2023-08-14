from os.path import join
import gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Beta
from torch.utils.data import DataLoader
from os import path
from time import sleep
import numpy
import collections
import math

from memory import Memory


device = torch.device("cuda")


class PPO:
    def __init__(
        self,
        env, #enironment
        net, #network model
        lr, #learning rate
        batch_size, 
        gamma, #discount future rewards
        gae_lambda, #uncertainty of advantges
        horizon,
        epochs_per_step,
        num_steps, #number of training steps
        clip, #epsilon
        value_coef, #discount critic loss
        entropy_coef, #discount entropy loss
        save_dir,
        save_interval,
    ):
        self.env = env
        self.net = net.to(device)

        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.horizon = horizon
        self.epochs_per_step = epochs_per_step
        self.num_steps = num_steps
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
        self.state = self.env.reset()
        self.state = self._fix_states(self.state)
        self.alpha = 1.0

    def _image_to_chwh(self, img):
        img = np.moveaxis(img, -1, 0)
        return img

    def _fix_states(self, state):
        state["image"] = self._image_to_chwh(state["image"])
        state["state"] = self._collect_navigation_info(self.env)
        lidar_raw = self.env.vehicle.lidar.perceive(self.env.vehicle)[0]
        for i in range(len(lidar_raw)):
            lidar_raw[i] = (1.01 - lidar_raw[i])
        state["lidar"] = lidar_raw
        return state

    def train(self):
        for step in range(self.num_steps):
            self._set_step_params(step)
            # collect episode trajectory for the horizon length
            with torch.no_grad():
                memory = self.collect_trajectory(self.horizon)

            memory_loader = DataLoader(
                memory, batch_size=self.batch_size, shuffle=True,
            )

            avg_loss = 0.0

            for epoch in range(self.epochs_per_step):
                for (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    advantages,
                    values,
                ) in memory_loader:
                    loss, _, _ = self.train_batch(
                        states, actions, log_probs, rewards, advantages, values
                    )
            
                    avg_loss += loss

            print("Loss", avg_loss / len(memory_loader))
            print("Step", step)

            if step % self.save_interval == 0:
                self.save(join(self.save_dir, f"net_{step}.pth"))

        # save final model
        self.save(join(self.save_dir, f"net_final.pth"))

    def train_batch(
        self,
        states,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
    ):
        self.optim.zero_grad()

        values, alpha, beta = self.net(states[0], states[1], states[2]) #feed nn
        values = values.squeeze(1) 

        policy = Beta(alpha, beta)
        entropy_loss = -policy.entropy().mean()
        log_probs = policy.log_prob(old_actions).sum(dim=1)

        ratio = (log_probs - old_log_probs).exp()  # same as policy / policy_old
        policy_loss_raw = ratio * advantages
        policy_loss_clip = (
            ratio.clamp(min=1 - self.clip, max=1 + self.clip) * advantages
        )
        policy_loss = -torch.min(policy_loss_raw, policy_loss_clip).mean() # maximize the rewards by minimizing losses

        with torch.no_grad():
            value_target = advantages + old_values  

        value_loss = nn.MSELoss()(values, value_target) # minimizing critic loss

        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        loss.backward()

        self.optim.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def collect_trajectory(self, num_steps: int, delay_ms: int = 0) -> Memory:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        for t in range(num_steps):
            tensor_img, tensor_info, tensor_lidar = self._split_state_tensor(self.state)

            #print("nav", tensor_info)
            value, alpha, beta = self.net(tensor_img,
                                          tensor_info,
                                          tensor_lidar)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)

            policy = Beta(alpha, beta)
            action = policy.sample()
            log_prob = policy.log_prob(action).sum()
            normalized_action = numpy.interp(action.cpu().numpy(),(0,1),(-1,1))
            next_state, reward, done, _ = self.env.step(normalized_action)

            if done:
                next_state = self.env.reset()

            next_state = self._fix_states(next_state)

            # store data
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            self.state = next_state

            self.env.render()

            if delay_ms > 0:
                sleep(delay_ms / 1000)

        # value of last state
        tensor_img, tensor_info, tensor_lidar = self._split_state_tensor(self.state)
        final_value, _, _ = self.net(tensor_img, tensor_info, tensor_lidar)
        final_value = final_value.squeeze(0)

        # generalized advantage estimates
        advantages = self._compute_gae(rewards, values, dones, final_value)

        #tensoring of states
        tensored_state_im = []
        tensored_state_inf = []
        tensored_state_lid = []
        for s in states:
            tensored_state_im.append(self._to_tensor(s["image"]))
            tensored_state_inf.append(self._to_tensor(s["state"]))
            tensored_state_lid.append(self._to_tensor(s["lidar"]))

        cated_dict = {"image": torch.cat(tensored_state_im),
                      "state": torch.cat(tensored_state_inf),
                      "lidar": torch.cat(tensored_state_lid)}

        # tensoring of data
        states = cated_dict
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        values = torch.cat(values)
        self.env.reset()
        return Memory(states, actions, log_probs, rewards, advantages, values)

    def save(self, filepath: str):
        torch.save(self.net.state_dict(), filepath)

    def _compute_gae(self, rewards, values, dones, last_value): # Generalized Advantage Estimation
        advantages = [0] * len(rewards)

        last_advantage = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (1 - dones[i]) * self.gamma * last_value - values[i]
            advantages[i] = (
                delta + (1 - dones[i]) * self.gamma * self.gae_lambda * last_advantage
            )

            last_value = values[i]
            last_advantage = advantages[i]

        return advantages

    def _to_tensor(self, x):
        conv = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        return conv

    def _set_step_params(self, step):
        # lower learning rate with each pass
        self.alpha = 1.0 - step / self.num_steps

        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr * self.alpha

    def _collect_navigation_info(self, env):
        car_navigation_info = (env.vehicle.navigation.get_navi_info()[0], #y relative to target
                               env.vehicle.navigation.get_navi_info()[1], #x telative to target
                               numpy.interp((env.vehicle.dist_to_left_side), (0, 30), (0, 1)),
                               numpy.interp((env.vehicle.dist_to_right_side), (0, 30), (0, 1)),
                               numpy.interp(abs(self.env.vehicle.velocity[0])+abs(self.env.vehicle.velocity[1]),(0,40), (0, 1)))
        return car_navigation_info
    def _split_state_tensor(self, state):
        tensor_image = self._to_tensor(state["image"])
        tensor_info = self._to_tensor(state["state"])
        tensor_lidar = self._to_tensor(state["lidar"])
        return tensor_image, tensor_info, tensor_lidar