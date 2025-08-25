"""
记得后续成立一个参数文件，便于后续开发修改参数！！！！！！！
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollection, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from typing import Tuple
from isaaclab.sensors import patterns, RayCasterCfg, ContactSensorCfg
# 无需手写 __init__ 方法，所有带类型注解的成员都会自动作为构造函数的参数并被初始化
@configclass
class H1robotEnvCfg(DirectRLEnvCfg):
    # 暂定渲染一次，就物理运行迭代一次
    decimation = 1

    # 每一个环境的持续的最长时间为1000s
    episode_length_s = 1000

    # action_scale: 动作的缩放因子，用于调整控制输入的幅度
    action_scale = 1

    # 动作有缩放时用的，这个其实有bug，暂时注释掉
    # action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    # 暂时不观测，先设置为0
    observation_space = 0
    state_space = 0

    # 一共19个动作空间
    action_space = 19

    data_path = "robot_project/H1robot/data/amass_all.pkl"

    device = "cuda"

    stack = "list"


    """
    机器人USD位置,此处我使用了绝对文件位置,若报错可切换相对文件位置,因为目前个模块都才开始配置
    文件的具体位置还需要调整，所以暂不使用相对位置
    请记得后续作调整

    首先，不使用带手的
    """

    H1_with_usd_path = "file:///home/ljk/IsaacLab/robot_project/H1robot/only_H1.usd"
 
    # 此处暂不定义，考虑到H1可能不存在渲染问题
    # 所以直接使用默认配置即可
    # render_cfg = sim_utils.RenderCfg(enable_translucency=True)

    """
    记得如果报错显示运算空间不足, GPU buffer overflow
    就要用下面的部分来修正,这里的API不是ISaaclab的API, 要去文档里面查询
    max_gpu_contact_pairs=2**23, default_buffer_size_multiplier=5,

    下面这个记得在刚体里面做调整
    max_depenetration_velocity=1.0,

    在Isaaclab已经移动到了单个刚体的配置里面
    contact_offset=0.01, rest_offset=0.0,
    """
    physxcfg=PhysxCfg(solver_type=1,
            min_position_iteration_count=1, max_position_iteration_count=8,
            min_velocity_iteration_count=0, max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5, 
            
        )

    # 只保留必要参数即可,目前的AMASS是30帧
    # 后续有需要可以调整，也可以把参数整合到一个文件里面
    sim: SimulationCfg = SimulationCfg(dt=1 / 30, render_interval=decimation, 
                                       physx=physxcfg , gravity=(0.0, 0.0, -9.81), 
                                       device="cuda")
    
    # H1机器人USD配置
    H1_robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path = "/World/envs/env_.*/H1Robot",
        spawn=sim_utils.UsdFileCfg(
        usd_path=H1_with_usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_joint": -0.52,  # -30 degrees
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle_joint": 20.0},
            damping={".*_ankle_joint": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
                },
            ),
            },
        )    

    # 创造的环境数量, 暂时只使用一个。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=20, replicate_physics=True)
    
    obs_context_len = 8  # 观测历史缓存长度，前一次帧信息累积。
    action_delay = 0  # 动作执行延迟帧数，-1 表示无延迟, 此处暂时设定为0,因为只播放和演示

    """
    下面都是未来可能用到的参数，暂时不改原值，也不使用
    """

    # 是否把高度阵列放入 obs（RayCaster 总是会更新，加入 obs 与否由此开关）
    measure_heights = False  
    # 控制射线采样网格的分辨率，也就是相邻采样点之间的间隔
    height_pattern_resolution = 0.1
    # 定义射线采样区域的总边界（x 方向和 y 方向的大小）
    # 表示你想在机器人基础框架下方 5cm × 5cm 范围上进行高度扫描
    height_pattern_size_xy = (0.05, 0.05)

    # Raycfg
    H1bot_Heightray: RayCasterCfg = RayCasterCfg(
        # prim_path="/World/envs/env_.*/H1Robot/base|/World/envs/env_.*/H1Robot/pelvis", 
        prim_path="/World/envs/env_.*/H1Robot/pelvis", 
        update_period=0.02,
        attach_yaw_only=True,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        pattern_cfg=patterns.GridPatternCfg(
        resolution=height_pattern_resolution, 
        size=height_pattern_size_xy
        ),
        debug_vis=False,
        # 扫描场景里什么部分
        mesh_prim_paths=["/World/ground"],
        )
    
    H1bot_term_contacts: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/H1Robot/(pelvis|.*hip.*|.*shoulder.*|.*elbow.*|.*knee.*)",
            update_period=0.0, history_length=4, debug_vis=False
        )
    H1bot_feet_contacts: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/H1Robot/.*ankle.*",  # 若资产命名为 FOOT/HEEL，请改正则
            update_period=0.0, history_length=6, debug_vis=False
        )

