import torch
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
parser.set_defaults(video=True)
parser.set_defaults(video_length=800)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from H1_env_cfg import H1robotEnvCfg
from H1_env import H1robotEnv

if __name__ == "__main__":
    env_cfg = H1robotEnvCfg()

    env = H1robotEnv(cfg = env_cfg)

    # 动作会被忽略（纯播放），随便给：
    actions2 = torch.tensor([[0.4]*19 + [0.001]], dtype=torch.float32)

    while simulation_app.is_running():
        obs, reward, terminated, truncated, info = env.step(action=actions2)

