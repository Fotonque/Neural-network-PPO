import torch
from torch import nn
import gym
from gym.spaces import Box
import numpy as np
from collections import deque

class RacingNet(nn.Module):
    def __init__(self, state_dim_images, state_dim_info, state_dim_lidar, action_dim) -> None:
        super().__init__()

        random_neurons = False

        #input dimensions
        n_actions = action_dim[0]
        n_navigations = state_dim_info[0]
        n_lidar = state_dim_lidar[0]

        #image manipulations
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim_images[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )


        conv_out_size = self._get_conv_out(state_dim_images)
        print(conv_out_size)
        n_encoder_out = 512

        if random_neurons == True:
           img_inc1 = int(np.random.randint(1024, 4096+1,size=(1)))
           n_encoder_out = int(np.random.randint(128, 1024+1,size=(1)))
           self.image_encoder = nn.Sequential(
                nn.Linear(conv_out_size, img_inc1), nn.ReLU(), nn.Linear(img_inc1, n_encoder_out), nn.ReLU(),
            )
        else:
            self.image_encoder = nn.Sequential(
                nn.Linear(conv_out_size, 2048), nn.ReLU(), nn.Linear(2048, n_encoder_out), nn.ReLU(),
            )

        #navigation manipulations
        n_navigations_out = 512
        if random_neurons == True:
           img_inc1 = int(np.random.randint(64, 256+1,size=(1)))
           n_navigations_out = int(np.random.randint(128, 1024+1,size=(1)))
           self.info = nn.Sequential(
                nn.Linear(n_navigations, img_inc1), nn.ReLU(), nn.Linear(img_inc1, n_navigations_out), nn.ReLU(),
            )
        else:
            self.info = nn.Sequential(
                nn.Linear(n_navigations, 128), nn.ReLU(), nn.Linear(128, n_navigations_out), nn.ReLU(),
            )

        #lidar manipulations
        n_lidar_out = 512
        if random_neurons == True:
           img_inc1 = int(np.random.randint(20, 128+1,size=(1)))
           n_lidar_out = int(np.random.randint(128, 1024+1,size=(1)))
           self.lidar = nn.Sequential(
                nn.Linear(n_lidar, img_inc1), nn.ReLU(), nn.Linear(img_inc1, n_lidar_out), nn.ReLU(),
            )
        else:
            self.lidar = nn.Sequential(
                nn.Linear(n_lidar, 20), nn.ReLU(), nn.Linear(20, n_lidar_out), nn.ReLU(),
            )

        combined_out_size = n_encoder_out + n_navigations_out + n_lidar_out
        print(combined_out_size)

        #actor's prediction
        self.actor_fc = nn.Sequential(nn.Linear(combined_out_size, 256), nn.ReLU(),)

        #beta-distibutions for action 
        self.alpha_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())

        # Estimates the value of the state
        self.critic = nn.Sequential(
            nn.Linear(combined_out_size, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, x, info, lidar):
        #neural inputs
        x = self.conv(x)
        x = self.image_encoder(x)
        info = self.info(info)
        lidar = self.lidar(lidar)

        #neural inputs combination
        combined_x_info = torch.cat((
            x.view(x.size(0), -1), 
            info.view(info.size(0), -1), 
            lidar.view(lidar.size(0),-1)), dim=1)

        # get value of the state
        value = self.critic(combined_x_info)

        # get value of actor
        combined_x_info = self.actor_fc(combined_x_info)

        # add 1 to alpha & beta to ensure correct distribution 
        alpha = self.alpha_head(combined_x_info) + 1
        beta = self.beta_head(combined_x_info) + 1

        return value, alpha, beta

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)

        return int(np.prod(x.size()))