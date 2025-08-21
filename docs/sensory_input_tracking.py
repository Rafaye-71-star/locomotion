# %% [markdown]
# # 轮次试运行期间 跟踪 感官输入和动作命令的 示例
# 
# 在轮次试运行期间，无论是否属于模型可观察到的物理量，访问和记录 Mujoco 物理模拟中涉及的任何物理量是微不足道的。


# %% [markdown]
# # Imports
import os
import platform
import requests
import zipfile

import numpy
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw
import mediapy
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
from acme import wrappers
from acme.tf import utils as tf2_utils

from flybody.download_data import figshare_download
from flybody.fly_envs import flight_imitation
# dm-reverb 不支持除 linux 以外的系统
if platform.system().lower() == 'linux':
    from flybody.agents.utils_tf import TestPolicyWrapper

# 防止 TensorFlow 窃取所有 GPU 显存。
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

render_kwargs = {'width': 640, 'height': 480}


# %% [markdown]
# ## 下载扑翼模式发生器（wing-beat pattern generator, WPG）基本模式和训练有素的飞行策略
# 
# 所需的数据已下载到本地`data`目录。 `flybody`补充数据也可以通过 <https://doi.org/10.25378/janelia.25309105> 访问。

# figshare_download(['flight-imitation-dataset', 'trained-policies'])
cur_dir = os.path.split(os.path.realpath(__file__))[0]

wpg_path = os.path.join(cur_dir, 'data/wing_pattern_fmech.npy')
ref_flight_path = os.path.join(cur_dir, 'data/flight-dataset_saccade-evasion_augmented.hdf5')
flight_policy_path = os.path.join(cur_dir, 'data/policy/flight')


# %% [markdown]
# # 创建飞行模仿任务环境
env = flight_imitation(
    ref_path=ref_flight_path,
    wpg_pattern_path=wpg_path,
    terminal_com_dist=float('inf'),
)
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

_ = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)


# %%
env.observation_spec()

env.action_spec()

# %% [markdown]
# # 加载预训练的飞行策略
policy = tf.saved_model.load(flight_policy_path)
if platform.system().lower() == 'linux':
    policy = TestPolicyWrapper(policy)


# %% [markdown]
# # 试运行一个轮次和记录：
# 1. 可观察的感觉输入
# 2. 动作命令
# 3. 物理模拟器中的未观察到的物理量（果蝇身体位置和朝向）
# 4. (也可以录制视频)

n_steps = 150

timestep = env.reset()

# 分配输入的内存
joints_pos = np.zeros((n_steps, 25))  # 本体感受：可观察的关节角度。
vel = np.zeros((n_steps, 3))  # 速度计
zaxis = np.zeros((n_steps, 3))  # 重力方向。
root_qpos = np.zeros((n_steps, 7))  # 果蝇的位置和朝向
actions = np.zeros((n_steps, 6))  # 翅膀动作命令。

frames = []

for i in tqdm(range(n_steps)):
    # 记录一些感官输入
    joints_pos[i] = timestep.observation['walker/joints_pos']
    vel[i] = timestep.observation['walker/velocimeter']
    zaxis[i] = timestep.observation['walker/world_zaxis']
    root_qpos[i] = env.physics.data.qpos[:7].copy()
    
    frames.append(env.physics.render(camera_id=1, **render_kwargs))

    # 模拟步进
    # 没有使用分布式策略，在这里补充TestPolicyWrapper里的一些操作
    batched_observation = tf2_utils.add_batch_dim(timestep.observation)
    distribution = policy(batched_observation)
    action = distribution.mean()  # 不是测试模式（测试模式返回均值和方差）
    if not (type(action) is numpy.float64):
        action = action[0, :].numpy()  # Remove batch dimension.
    timestep = env.step(action)

    # 记录翅膀的动作命令
    actions[i] = action[3:9]


# %% [markdown]
# ## 显示首次展示视频
import cv2
import numpy as np

def display_video(frames, fps=3):
    window_name = 'Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame in frames:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# 将“mediapy”（交互模式）替换为“OpenCV”用于视频在脚本模式中显示。
display_video(frames, fps=3)
# 交互式环境中使用 mediapy 显示视频
# mediapy.show_video(frames)


# %%
# 打印可观察的关节名称和索引。
[(i, joint.name) for i, joint in enumerate(env.task.walker.observable_joints)]

# # 绘制首次展示的示例感官输入
# ### 本体感受感官输入：头部，腹部，翅膀关节角度
time_axis = np.arange(n_steps) * env.control_timestep() * 1000  # ms

plt.figure(figsize=(6, 10))
plt.suptitle('Proprioception sensory inputs: head, abdomen, wing joint angles')
plt.subplot(4, 1, 1)  # 头部关节
plt.plot(time_axis, joints_pos[:, :3], label=['head_abduct', 'head_twist', 'head'])
plt.ylabel('Head joint angles (rad)')
plt.legend()
plt.subplot(4, 1, 2)  # 腹部关节。
plt.plot(time_axis, joints_pos[:, 9:23])
plt.ylabel('Abdomen joint angles (rad)')
plt.subplot(4, 1, 3)  # 左翅
plt.plot(time_axis, joints_pos[:, 3:6], label=['yaw', 'roll', 'pitch'])
plt.ylabel('Left wing angles (rad)')
plt.legend()
plt.subplot(4, 1, 4)  # 右翅
plt.plot(time_axis, joints_pos[:, 6:9])
plt.xlabel('Time (ms)')
plt.ylabel('Right wing angles (rad)')
plt.tight_layout()


# %% [markdown]
# ### 以自我为中心的速度向量
plt.plot(time_axis, vel, label=['vx', 'vy', 'vz'])
plt.xlabel('Time (ms)')
plt.ylabel('x,y,z components of egocentric\nvelocity vector (cm/s)')
plt.legend()


# %% [markdown]
# ### 以自我为中心的方向向量
plt.plot(time_axis, zaxis, label=['x', 'y', 'z'])
plt.xlabel('Time (ms)')
plt.ylabel('x,y,z components of egocentric\ngravity direction unit vector')
plt.legend()


# %% [markdown]
# ### 翅膀动作命令
# 无单位, 在 (-1, 1) 之间
plt.subplot(2, 1, 1)
plt.plot(time_axis, actions[:, :3], label=['yaw', 'roll', 'pitch'])
plt.ylabel('Left wing control\n(unitless)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time_axis, actions[:, 3:])
plt.xlabel('Time (ms)')
plt.ylabel('Right wing control')


# %% [markdown]
# ### 全局坐标中的果蝇身体位置和朝向

# 除了直接可观察到的果蝇模型之外，可以直接从模拟器访问任何其他与模拟相关的物理量。
# 
# Mujoco物理模拟的完整状态封装在单个`MJDATA`数据架构中。有关更多信息，请参见：
# </br>
# https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata

plt.subplot(2, 1, 1)
plt.plot(time_axis, root_qpos[:, :3], label=['x', 'y', 'z'])
plt.ylabel('x,y,z position of root joint\nin world coordinates (cm)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time_axis, root_qpos[:, 3:], label=['w', 'x', 'y', 'z'])
plt.xlabel('Time (ms)')
plt.ylabel('Components of root joint\nquaternion in world\ncoordinates (unitless)');




