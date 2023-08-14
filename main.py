import argparse
import random

from metadrive.policy.idm_policy import IDMPolicy
from ppo import PPO
from nncar import RacingNet
import gym
import torch

from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel
from metadrive import MetaDriveEnv


STEPS = 125
EpS = 4
BATCH = 128

GAMMA = 0.99
GAE = 0.95
CLIP = 0.1

LR = 1e-7
value_coef = 0.5
entropy_coef = 0.01

save_dir = 'ckpt'
save_interval = 5


PATH = "ckpt/finetuning3.pth"

# class CarRacing(gym.Wrapper):
#     def __init__(self):
#         config = dict(
#             # controller="joystick",
#             use_render=True,
#             manual_control=False,
#             traffic_density=0.1,
#             num_scenarios=100,
#             random_agent_model=True,
#             random_lane_width=True,
#             random_lane_num=True,
#             #use_lateral_reward=True,
#             # debug=True,
#             # debug_static_world=True,
#             map=4,  # seven block
#             start_seed=random.randint(0, 1000)
#         )
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
#         args = parser.parse_args()
#         if args.observation == "rgb_camera":
#             config.update(dict(image_observation=True))
#             config.update(dict(image_on_cuda=True))
#         self.env = MetaDriveEnv(config)
#         super().__init__(self.env)

if __name__ == "__main__":

    #env = CarRacing()
    env = MetaDriveEnv(
        {
            "num_scenarios": 100,
            "traffic_density": 0.10,
            "debug": False,
            # "global_light": False,
            "image_observation": True,
            # "controller": "joystick",
            "manual_control": False,
            "use_render": False,
            "accident_prob": 0,
            "decision_repeat": 5,
            "interface_panel": [MiniMap, VehiclePanel, RGBCamera],
            "need_inverse_traffic": False,
            "rgb_clip": True,
            #"map": "OXX",
            #"map": "OXSS",
            #"agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": False,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "success_reward": 50,
            #"out_of_road_penalty": 10,
            #"crash_vehicle_penalty": 10,
            #"crash_object_penalty": 10,
            "force_destroy": False,
            "vehicle_config": {
                "enable_reverse": False,
                "rgb_to_grayscale": True,
                "vehicle_model": "s",
                "rgb_camera": (150, 150),
                "spawn_velocity": [3.728615581032535, -0.04411703918728195],
                "spawn_velocity_car_frame": True,
                "show_lidar": False,
                "spawn_lane_index": None,
                # "destination":"2R1_3_",
                "show_side_detector": False,
                "show_lane_line_detector": False,
                "side_detector": dict(num_lasers=2, distance=50),
                "lane_line_detector": dict(num_lasers=2, distance=50),
                "show_line_to_navi_mark": False,
                "show_navi_mark": False,
                "show_dest_mark": False
            },
        }
    )
    state_dim_image = (3,150,150)
    state_dim_info = (5,1)
    state_dim_lidar = (240,1)
    actions_dim = (2,1)
    net = RacingNet(state_dim_image, state_dim_info, state_dim_lidar, actions_dim)
    try:
        net.load_state_dict(torch.load(PATH))
    except Exception as e:
        net = torch.load(PATH)

    ppo = PPO(
        env,
        net,
        LR,
        BATCH,
        GAMMA,
        GAE,
        1024,
        EpS,
        STEPS,
        CLIP,
        value_coef,
        entropy_coef,
        save_dir,
        save_interval,
    )

    ppo.train()

    env.close()
