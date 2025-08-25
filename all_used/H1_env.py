
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from typing import Dict, Optional, List, Tuple
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollection, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.sensors import patterns, RayCaster, ContactSensor
import numpy as np

import tool_func
# 此处调用H1_env_cfg.py脚本，引入里面的H1robotEnvCfg类
from H1_env_cfg import H1robotEnvCfg

class H1robotEnv(DirectRLEnv):
    cfg: H1robotEnvCfg

    def __init__(self, cfg: H1robotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.H1robot = self.scene["H1robot"]
        self.num_dofs = self.H1robot.num_joints
        # 记录环境数
        nenv = self.num_envs
        # 用 obs_buf 来存储每个环境当前的“进度”信息，构建buffer缓冲区
        self.obs_buf = torch.zeros(nenv, 2, device=self.device)
        # 用于存储每个环境实例（nenv 个并行环境）当前步的奖励值
        self.rew_buf = torch.zeros(nenv, device=self.device)
        # 标记哪些环境需要被重置
        self.reset_buf = torch.ones(nenv, dtype=torch.long, device=self.device)
        # 标识每个环境是否因为“时长耗尽”（timeout）而需要被重置
        self.time_out_buf = torch.zeros(nenv, dtype=torch.bool, device=self.device)
        # 记录每个环境当前 episode 已执行的步数
        self.episode_length_buf = torch.zeros(nenv, dtype=torch.long, device=self.device)
        # 计数器，用于记录所有环境一共执行了多少步（全局计步），这个可有可无吧，暂时保留在这
        self.common_step_counter = 0
        # 读取资产的默认关节角，作为 dof_offset（可选） ===
        self.default_pos = self.H1robot.data.default_joint_pos.clone().to(self.device)
        if self.default_pos.numel() == 0:
            # 某些资产可能未写默认值，兜底为 0
            self.default_pos = torch.zeros(self.num_dofs, device=self.device)

        self.action_scale = self.cfg.action_scale
        self._init_target_jt()

        # 每个 env 一个序列 idx；frame_idx 表示当前帧；长度 length_env 为每 env 的有效长度（由 mask/lengths 共同确定）
        # self.seq_idx = torch.randint(0, self.B, (nenv,), device=self.device)        # [nenv]
        self.seq_idx = (torch.arange(nenv, device=self.device) % self.B).long()     # [nenv] 0,1,2,... 顺序分配
        self.frame_idx = torch.zeros(nenv, dtype=torch.long, device=self.device)    # [nenv]
        # 计算每个 env 的“有效长度”：优先用 mask 的最后 True 位置 +1；回退到 lengths_all[seq]
        self.length_env = self._calc_length_env(self.seq_idx)                       # [nenv]

        # === 每个 env 的节拍控制（支持不同 fps）===
        # traj_dt = 1/fps；将其换算为仿真步数（浮点），再随机取整为倒计时 tick。
        self.traj_dt_env = 1.0 / self._fps_of(self.seq_idx)                         # [nenv]
        self.traj_steps_env = self.traj_dt_env / self.sim.cfg.dt                        # [nenv] float
        self.tick_left = tool_func.sample_int_from_float(self.traj_steps_env)                    # [nenv] int >=1

        # 初始化：reset 全部 env，立刻写 0 帧
        self._reset_idx(torch.arange(nenv, device=self.device))
        self._write_cur_frame_states()



    def _setup_scene(self):

        # 地面（可替换为 TerrainImporterCfg，演示先用标准 ground plane）
        sim_utils.spawn_ground_plane("/World/ground", GroundPlaneCfg(),
                                     translation = (0.0, 0.0, -1.8))

        # 机器人
        self.scene.articulations["H1robot"] = Articulation(self.cfg.H1_robot_cfg)
       # 高度扫描 RayCaster（附着在“骨盆/基座”，只随 yaw 对齐），暂时不使用
        self.H1_height_scanner = RayCaster(self.cfg.H1bot_Heightray)

        # 两个接触传感器，一个骨盆传感器，一个脚踝传感器
        self.H1_term_contacts = ContactSensor(self.cfg.H1bot_term_contacts)
        self.H1_feet_contacts = ContactSensor(self.cfg.H1bot_feet_contacts)

        self.scene.sensors["height_scanner"] = self.H1_height_scanner
   
        # 终止接触传感器（骨盆/大关节等）
        self.scene.sensors["term_contacts"] = self.H1_term_contacts
        # 脚/踝接触（步态与接触罚项可用）
        self.scene.sensors["feet_contacts"] = self.H1_feet_contacts
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 必须 reset 才会创建好句柄/缓冲，暂时先不用
        # self.scene.reset()
    
    def _init_target_jt(self):
        # 先按 default_qpos 做偏移.
        # 注意 default_qpos shape [dof]，与 npy [*, dof] 对齐
        """
        tensor([[ 0.0000,  0.0000, -0.3490,  0.6980, -0.3490,  0.0000,  0.0000, -0.3490,
          0.6980, -0.3490,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000]], device='cuda:0')  
        """
        batch, lengths, meta = tool_func.load_all_from_any(
            device=self.device,
            path=self.cfg.data_path,
            dof_offset=self.default_pos ,                 # 纯回放 -> 绝对角，不叠加 offset
            stack=self.cfg.stack,
        )
        self.names: List[str] = batch["names"]
        self.root_trans_all: torch.Tensor = batch["root_trans"]        # [B,T,3]
        self.root_rot_all: torch.Tensor = batch["root_rot"]            # [B,T,4]
        self.dof_all: torch.Tensor = batch["dof"]                      # [B,T,D]
        self.mask_all: torch.Tensor = batch["mask"].bool()             # [B,T]
        self.fps_all: torch.Tensor = batch["fps"].to(torch.float32)    # [B]
        self.lengths_all: torch.Tensor = lengths                       # [B] 原始有效长度（未对齐前）

        self.B, self.Tmax, self.D = self.dof_all.shape
        assert self.D == self.num_dofs, f"数据 DOF={self.D} 与资产 DOF={self.num_dofs} 不一致"
        

    def _pre_physics_step(self, actions: torch.tensor) -> None:
        # self.actions = self.action_scale * actions.clone()
        # 只记录步数即可
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.post_physics_step()

    
    def post_physics_step(self):
        # 逐 env 递减 tick，<=0 时推进一帧并重采样 tick
        self.tick_left -= 1
        advance_mask = self.tick_left <= 0
        if torch.any(advance_mask):
            self._advance_frame(advance_mask.nonzero(as_tuple=False).flatten())
            self._write_cur_frame_states()
            # 为这些 env 重新采样下一帧的 tick
            self.tick_left[advance_mask] = tool_func.sample_int_from_float(self.traj_steps_env[advance_mask])

        # 需要 reset 的 env
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self._reset_idx(env_ids)

    def _apply_action(self) -> None:
        """"""


    def _get_observations(self) -> dict:
        return {}


    def get_rewards(self) -> torch.Tensor:
        return self.rew_buf
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 终止条件（非循环回放时，到达序列末帧就终止）
        self.loop_playback = True
        at_end = self.frame_idx >= (self.length_env - 1)          # [nenv] bool
        resets = (not self.loop_playback) & at_end               # 仅在不循环时作为真正的 "done"

        # 超时条件（由框架维护的 episode_length_buf 与 max_episode_length
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        # 回 (resets, time_out)，形状均为 [nenv] 的 bool Tensor
        return resets.bool(), time_out.bool()


    def get_extras(self) -> Dict:
        return {}

    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(1,1)
            
        return total_reward



    # def _reset_idx(self, env_ids: Sequence[int] | None):
    #     """"""

    #     # joint_pos[:, self._pole_dof_idx] += sample_uniform(
    #     #     self.cfg.initial_pole_angle_range[0] * math.pi,
    #     #     self.cfg.initial_pole_angle_range[1] * math.pi,
    #     #     joint_pos[:, self._pole_dof_idx].shape,
    #     #     joint_pos.device,
    #     # )


    def compute_rewards(self):
        total_reward = 0
        return total_reward
    
    def _fps_of(self, seq_idx: torch.Tensor) -> torch.Tensor:
        # 返回选中序列的 fps（float）
        fps = self.fps_all[seq_idx].clamp(min=1.0)  # 防止 0
        return fps

    def _calc_length_env(self, seq_idx: torch.Tensor) -> torch.Tensor:
        # 基于 mask 计算有效长度（True 的最后位置 +1），若没有 True 就回退到 lengths_all
        B, T = self.mask_all.shape
        mask_sel = self.mask_all[seq_idx]            # [nenv, T]
        any_true = mask_sel.any(dim=1)               # [nenv]
        # 默认长度=lengths_all[seq] 与 Tmax 的最小值（避免越界）
        length = torch.minimum(self.lengths_all[seq_idx], torch.tensor(T, device=self.device, dtype=torch.long))
        # 对存在有效 mask 的 env，用 mask 的最后 True 位置+1
        if any_true.any():
            idxs = any_true.nonzero(as_tuple=False).flatten()
            last_true = mask_sel[idxs].float().argmax(dim=1)  # 注意：argmax 不是“最后 True”；我们需要反向搜索
            # 正确做法：从后向前找 True
            rev = torch.flip(mask_sel[idxs], dims=[1])        # [k, T]
            last_from_end = rev.float().argmax(dim=1)         # 第一个 True 的反向索引
            last_true_pos = (T - 1) - last_from_end
            length[idxs] = last_true_pos + 1
        # 长度至少为 1
        length = length.clamp(min=1)
        return length
    
    def _advance_frame(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.frame_idx[env_ids] += 1
        cur_len = self.length_env[env_ids]
        over = self.frame_idx[env_ids] >= cur_len

        if True:
            if over.any():
                # 到尾回 0；可选择是否换序列
                self.frame_idx[env_ids[over]] = 0
                if True:
                    self._resample_seq(env_ids[over])
        else:
            # 不循环：设置 reset 标记，并 clamp 到最后一帧（下个循环会 reset）
            self.reset_buf[env_ids[over]] = 1
            self.frame_idx[env_ids] = torch.minimum(self.frame_idx[env_ids], cur_len - 1)

    def _resample_seq(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        # 随机选新序列
        # new_idx = torch.randint(0, self.B, (env_ids.numel(),), device=self.device)
        # self.seq_idx[env_ids] = new_idx
        self.seq_idx[env_ids] = (self.seq_idx[env_ids] + 1) % self.B
        # 该 env 的有效长度与 fps/节拍需要更新
        self.length_env[env_ids] = self._calc_length_env(self.seq_idx[env_ids])
        self.traj_dt_env[env_ids] = 1.0 / self._fps_of(self.seq_idx[env_ids])
        self.traj_steps_env[env_ids] = self.traj_dt_env[env_ids] / self.sim.cfg.dt
        # tick 在调用处会重采样

    def _write_cur_frame_states(self):
        # 选取每个 env 的当前序列与帧
        print(f"[DEBUG] seq_idx: {self.seq_idx.cpu().numpy()}, frame_idx: {self.frame_idx.cpu().numpy()}")
        env_idx = torch.arange(self.num_envs, device=self.device)
        s = self.seq_idx
        f = self.frame_idx

        root_pos = self.root_trans_all[s, f]     # [nenv,3]
        root_quat = self.root_rot_all[s, f]      # [nenv,4] (xyzw)
        q_ref = self.dof_all[s, f]               # [nenv,D]

        # if True:
        #     root_quat = torch.nn.functional.normalize(root_quat, dim=-1, eps=1e-12)

        root_state = torch.cat([root_pos, root_quat, 
                                torch.zeros_like(root_pos), 
                                torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
        # 直接写状态（纯回放）
        # 导出的是(x, y, z, w),但是isaacsim当中是(w, x, y, z)
        root_quat_xyzw = self.root_rot_all[s, f]                  # [nenv,4] (xyzw)
        root_quat_wxyz = torch.cat([root_quat_xyzw[..., 3:4],     # w
                            root_quat_xyzw[..., :3]], -1) # x,y,z
        root_state = torch.cat([root_pos, root_quat_wxyz,
                        torch.zeros_like(root_pos),
                        torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
        self.H1robot.write_root_state_to_sim(
            root_state = root_state

        )
        self.H1robot.write_joint_state_to_sim(q_ref, torch.zeros_like(q_ref))
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        # 计数归零
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.time_out_buf[env_ids] = 0

        # （可选）重采样序列
        if True:
            self._resample_seq(env_ids)

        # 帧归零
        self.frame_idx[env_ids] = 0

        # 为这些 env 初始化节拍 tick
        self.tick_left[env_ids] = tool_func.sample_int_from_float(self.traj_steps_env[env_ids])

        # 写入第 0 帧
        self._write_cur_frame_states()
