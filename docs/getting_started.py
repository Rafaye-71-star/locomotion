# # 开始使用“fly body”
# 
# `flybody'是一种果蝇的解剖学详细体系模型_ drosophila melanogaster_用于Mujoco物理模拟器和强化学习（RL）应用。 Fly模型是在Google DeepMind和HHMI Janelia Research Camp的合作努力中开发的。我们设想我们的模型是果蝇生物物理学模拟的平台，并在体现的环境中建模对感觉运动行为的神经控制。
# 
# 本笔记本显示了使用`dm_control'的Mujoco飞行模型操作的几个Python示例。
# 
# 有关更多背景信息，请探索：
# 
# * [MuJoCo 文档](https://mujoco.readthedocs.io/en/stable/overview.html)
# * [MuJoCo](https://github.com/google-deepmind/mujoco) and tutorials therein
# * [dm_control](https://github.com/google-deepmind/dm_control) and tutorials therein
# * dm_control [paper](https://arxiv.org/abs/2006.12983)
# * [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md)

# %% [markdown]
# ## 在colab上安装“飞body”
# ### _如果在本地运行笔记本，请跳过此单元！_

# %%
# 如果在Colab中，请运行此单元格安装 fly body。不要忘记选择GPU！
# 否则，如果本地运行笔记本，请跳过此单元格。

# import os
# import subprocess
# if subprocess.run('nvidia-smi').returncode:
#   raise RuntimeError(
#       'Cannot communicate with GPU. '
#       'Make sure you are using a GPU Colab runtime. '
#       'Go to the Runtime menu and select Choose runtime type.')

# 添加一个ICD配置，以便GLVND可以拾取NVIDIA EGL驱动程序。
# 这通常是作为NVIDIA驱动程序包的一部分安装的，但是Colab内核不会通过APT安装其驱动程序，因此ICD丢失了。
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
# NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
# if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
#   with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
#     f.write("""{
#     "file_format_version" : "1.0.0",
#     "ICD" : {
#         "library_path" : "libEGL_nvidia.so.0"
#     }
# }
# """)

# !pip install --upgrade pip
# 安装 flybody（必须）
# !python -m pip install git+https://github.com/TuragaLab/flybody.git

# 配置Mujoco以使用EGL渲染后端（需要GPU）。
# %env MUJOCO_GL=egl

# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw

from dm_control import mujoco
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

import flybody
from flybody.fly_envs import walk_on_ball
from flybody.utils import display_video, any_substr_in_str


# %% [markdown]
# 渲染中 帧尺寸和相机的 名字到id的映射
frame_size = {'width': 640, 'height': 480}
cameras = {'track1': 0, 'track2': 1, 'track3': 2,
           'back': 3, 'side': 4, 'bottom': 5, 'hero': 6,
           'eye_right': 7, 'eye_left': 8}

# # 独立飞行模型（在RL环境之外）

# %% [markdown]
# ## 加载Mujoco飞行模型
# 让我们加载 Fly Model XML 文件 `fruitfly.xml`，将其直接编译为`physics`对象，然后打印一些模型的参数。请注意，除了苍蝇本身外，身体的数量还包括一个额外的 worldbody，关节还包括一个自由接头 freejoint，将苍蝇与世界机构连接起来，自由接头 freejoint 又贡献了六个额外的自由度（3平移，3个旋转）。
flybody_path = os.path.dirname(flybody.__file__)
xml_path = os.path.join(flybody_path, 'fruitfly/assets/fruitfly.xml')

physics = mjcf.Physics.from_xml_path(xml_path)  # 加载和编译。

print('# of bodies:', physics.model.nbody)
print('# of degrees of freedom:', physics.model.nv)
print('# of joints:', physics.model.njnt)
print('# of actuators:', physics.model.nu)
print("fly's mass (gr):", physics.model.body_subtreemass[1])

# %% [markdown]
# ## 加载时可视化苍蝇
# 作为初始化的，苍蝇处于其默认的休息姿势，其中所有关节角度存储在`physics.data.qpos`中的关节角均设置为零。 `physics.data.qpos`是对基础Mujoco的`mjData->qpos`的数据结构的视图，该数据结构具有该模型的广义坐标。请注意，条目`qpos [：3]`对应于世界坐标中root自由接连接的笛卡尔XYZ位置，而`qpos [3：7]`是Quaternion，最初是代表根关节取向的Quaternion，最初设置为单位四元素`[1, 0, 0, 0]`，其余元素`qpos [7：]`代表了我们飞行模型的所有铰链关节的关节角度。

physics.data.qpos

# %% [markdown]
# 可视化不同摄像机的几种视角：“英雄”和“底部”相机
pixels = physics.render(camera_id=cameras['hero'], **frame_size)
PIL.Image.fromarray(pixels)

pixels = physics.render(camera_id=cameras['bottom'], **frame_size)
PIL.Image.fromarray(pixels)


# %% 
# 我们可以隐藏外部装门面的网格，
# 并揭露以蓝色显示的碰撞几何体原语。
# 另外，请注意，在飞行模拟（橙色）期间，用于流体（空气）相互作用的紫色和椭圆形的翅膀中显示的黏性几何体 geoms。
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.geomgroup[1] = 0  # 隐藏外部网格。
scene_option.geomgroup[3] = 1  # 使流体互动交互的椭圆形翅膀可见（橙色）。
scene_option.geomgroup[4] = 1  # 使碰撞几何体 geoms 可见（蓝色）。
pixels = physics.render(camera_id=cameras['side'], **frame_size, scene_option=scene_option)
PIL.Image.fromarray(pixels)


# %%
# ## 装载地板并可视化
# 我们还可以加载飞行模型与简单的平坦地板（和一个天箱 skybox），以进行更有意义的模拟

xml_path_floor = os.path.join(flybody_path, 'fruitfly/assets/floor.xml')
physics = mjcf.Physics.from_xml_path(xml_path_floor)

pixels = physics.render(camera_id=cameras['side'], **frame_size)
PIL.Image.fromarray(pixels)


# %% [markdown]
# ## 运动学操纵
# 让我们尝试使用飞行模型的一些简单的运动学操作，即，在没有计算任何力的情况下，我们将身体置于几个姿势中。
# 
# 例如，我们可以从上一个图像中的默认休息位置开始，并以一定角度将苍蝇围绕Z轴（垂直轴，指向）。这将需要更新存储在 `qpos[3:7]` 的 根关节四元素。
# 
# 为了进行运动学操作，除了写入`qpos`外，我们还必须在 Mujoco 的`mjData`数据结构中更新与位置相关的其他条目。一种方法是使用`physics.reset_context()`（有关更多详细信息请参阅[dm_control 论文](https://arxiv.org/abs/2006.12983) ）。

# %%
# 构建新的根关节四元素：围绕Z轴旋转90度。
angle = np.pi / 2
quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
# 写入 qpos 并更新 mjData 中的其他相关数量。
with physics.reset_context():
    physics.named.data.qpos[3:7] = quat
pixels = physics.render(camera_id=cameras['side'], **frame_size)
PIL.Image.fromarray(pixels)

# %% [markdown]
# 现在，我们可以尝试一些更多。
# 
# 让我们再次从默认的休息位置开始，然后尝试折叠翅膀。在模型中，每个机翼通过三个铰链关节 (yaw, roll, pitch) 连接到胸部，有效地代表了3度自由的球形关节。我们需要找到与折叠翅膀置相对应的翅膀关节角。这可以如下完成。
# 
# 除了执行器外，大多数果蝇关节都附有较弱的弹簧。在没有驱动或外力时，这些弹簧将作用于特定的预定义位置，例如折叠的翅膀、进行飞行时缩回的腿、保持长鼻缩回等。在每种情况下，这些关节角度存储在 XML文件的 关节 `springref` 属性中，将 MuJoCo `mjdata`数据结构 中的 `qpos_spring`属性存储 ，通过 `dm_control` 以`physics.model.qpos_spring'的方式暴露在这里。
# 
# 让我们找到翅膀关节，并从弹簧参数中读取折叠翅膀的关节角度：

# %%
wing_joints = []
folded_wing_angles = []
# 在所有模型关节上循环。
for i in range(physics.model.njnt):
    joint_name = physics.model.id2name(i, 'joint')
    # 如果存在翅膀关节，存储关节名称和参考角度。
    if 'wing' in joint_name:
        wing_joints.append(joint_name)
        folded_wing_angles.append(
            physics.named.model.qpos_spring[joint_name].item())

wing_joints, folded_wing_angles

# %% [markdown]
# 现在，我们可以将这些翅膀角度写入`qpos`中的相应字段。请注意，我们现在使用命名（和矢量化）索引将翅膀角度写入`qpos`。

with physics.reset_context():
    physics.named.data.qpos[wing_joints] = folded_wing_angles

pixels = physics.render(camera_id=cameras['side'], **frame_size)
PIL.Image.fromarray(pixels)

# %% [markdown]
# 以同样的方式，我们可以（运动方式）将腿放入缩回的飞行位置，该位置存储在`yphysics.model.qpos_spring`中，就像以前：

with physics.reset_context():
    # 遍历所有关节
    for i in range(physics.model.njnt):
        name = physics.model.id2name(i, 'joint')
        # 如果有腿关节，将关节角度设置为等于其相应的参考弹簧角。
        if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus'], name):
            physics.named.data.qpos[name] = physics.named.model.qpos_spring[name]
    # 另外，通过设置果蝇的根关节的 Z 坐标来抬起苍蝇。
    physics.data.qpos[2] = 1.  # 在这里，单位：cm。

pixels = physics.render(camera_id=cameras['side'], **frame_size)
PIL.Image.fromarray(pixels)

# %% [markdown]
# ## 运动学重放：折叠翅膀
# 运动式姿势的序列可用于运动式重放动画。
# 
# 在此示例中，我们将翅膀逐渐将其默认位置移至以前获得的折叠位置。在每个步骤中，我们都会渲染一个帧，最后生成一个从帧序列中生成的视频：

n_steps = 150
frames = []
for i in range(n_steps):
    with physics.reset_context():
        wing_angles = np.array(folded_wing_angles) * np.sin(np.pi/2 * i/n_steps)
        physics.named.data.qpos[wing_joints] = wing_angles
    pixels = physics.render(camera_id=cameras['back'], **frame_size)
    frames.append(pixels)

display_video(frames)

# %% [markdown]
# ## 模拟物理：程序驱动的身体运动
# 
# 现在，我们可以尝试运行实际的物理模拟。
# 
# 在此示例中，我们将以程序控制果蝇的执行器，并及时进行模拟，以生成一系列运动。
# 首先，让我们准备一组执行器名称以在模拟的每个阶段控制。
# 按照惯例，我们模型中的所有联合执行器（与肌腱执行器相反）的名称与它们所扮演的关节相同。

def get_leg_actuator_names(leg):
    return [f'{joint}_{leg}' 
            for joint in ['coxa', 'femur', 'tibia', 'tarsus', 'tarsus2']]

# 腿部执行器的名称。
leg_actuators_L1 = get_leg_actuator_names('T1_left')
leg_actuators_R1 = get_leg_actuator_names('T1_right')
leg_actuators_L2 = get_leg_actuator_names('T2_left')

# 前腿和中腿的联合运动振幅。
amplitude_front = 0.5 * np.array([1, -1, 2, 1, 1])
amplitude_mid = 0.5 * np.array([0.5, -0.5, -2, 1, 2])


# %% [markdown]
# Now we can simulate the motion sequence. At each control step, we will be writing the control into MuJoCo's `mjData->ctrl`, exposed here as `physics.data.ctrl`. We will also engage the leg adhesion actuators at a certain point during the motion sequence. 
# 
# The control semantics is the target joint angles for position actuators, and (scaled) force for force and adhesion actuators (see [MuJoCo docs](https://mujoco.readthedocs.io/en/stable/overview.html) for more details). With the exception of wings and adhesion, our model uses position actuators. The wings are powered by force (torque) actuators.
# 
# We will change the control once every `physics_to_ctrl_ratio == 10` simulation steps, to ensure simulation stability (see `dm_control` [paper](https://arxiv.org/abs/2006.12983) for more details).
# 
# Note that in the first part of the motion sequence, "Let wings fold", we don't alter `physics.data.ctrl` yet and it is still zero after resetting the simulation with `physics.reset()`. Nevertheless, the wings will fold -- this is achieved by the weak springs acting to move the wings to a reference position, as described above.

n_steps = 100
physics_to_ctrl_ratio = 10
frames = []

# Reset physics to initial default state.
physics.reset()

# Let wings fold.
for i in range(n_steps):
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()

# Twist head.
for i in range(n_steps):
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)
    physics.named.data.ctrl['head_twist'] = 1.5 * np.sin(2*np.pi * i/n_steps)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()

# Move middle right leg.
for i in range(n_steps+50):
    if i <= n_steps:
        physics.named.data.ctrl[leg_actuators_L2] = amplitude_mid * np.sin(np.pi * i/n_steps)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)

# Activate middle leg adhision to prevent slipping when front legs are lifted later.
physics.named.data.ctrl[['adhere_claw_T2_right', 'adhere_claw_T2_left']] = 1.

# Lift fronts legs with lag.
for i in range(n_steps+100):
    left_angle = np.pi * i/n_steps
    right_angle = np.pi * i/n_steps - np.pi/5
    if left_angle <= np.pi:
        physics.named.data.ctrl[leg_actuators_L1] = amplitude_front * np.sin(left_angle)
    if 0 < right_angle <= np.pi:
        physics.named.data.ctrl[leg_actuators_R1] = amplitude_front * np.sin(right_angle)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)

display_video(frames)

# %% [markdown]
# ## Actuate entire body with random actions
# Now let's actuate all of the degrees of freedom at once with random control.
# 
# As force actuators (wings) and position actuators (the rest of the body) have different control semantics (scaled force and target joint angles, respectively), we'll actuate them with control signals of different magnitude. Let's find the indices for each actuator group first:

# %%
wing_act_indices = []  # Force actuators.
body_act_indices = []  # Position actuators.
# Loop over all actuators.
for i in range(physics.model.nu):
    name = physics.model.id2name(i, 'actuator')
    # Store wing actuator indices and rest of indices separately.
    if 'wing' in name:
        wing_act_indices.append(i)
    else:
        body_act_indices.append(i)

print(wing_act_indices)
print(body_act_indices)

# %% [markdown]
# Run simulation and actuate the fly body with random actions.

# %%
n_body_actions = len(body_act_indices)
n_wing_actions = len(wing_act_indices)

n_steps = 300
physics_to_ctrl_ratio = 10
frames = []

# Reset simulatiomn to initial default state.
physics.reset()

for i in range(n_steps):
    pixels = physics.render(camera_id=cameras['side'], **frame_size)
    frames.append(pixels)
    physics.named.data.ctrl[body_act_indices] = np.random.uniform(-.3, .3, n_body_actions)
    physics.named.data.ctrl[wing_act_indices] = np.random.uniform(-.005, .005, n_wing_actions)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()

display_video(frames)

# %% [markdown]
# ## Model modifications: adding leg MoCap sites using [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md)
# 
# In addition to manipulating joint angles and controls, as we did above, the fly model itself can be modified. All aspects of the model (e.g., sizes of body parts, actuator strengths and types, degrees of freedom, masses, appearance, etc.) can be easily changed programmatically.
# 
# Let's consider a simple example of fly body modifications. Assume we've got a motion capture dataset tracking the positions of the fly leg joints and we would like to fit the fly model's leg poses to this data. One way of doing so will require referencing the corresponding keypoint positions in the fly legs, which in turn can be achieved by adding sites to the leg joints.
# 
# In contrast to loading and compiling the model to a `physics` object in one step as we did before, we will split this process into two steps. First, we will load the fly model as `mjcf_model`, a python object model for the underlying MuJoCo fly XML file, which we can interact with and modify programmatically (see `dm_control`'s [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md)). Second, we will compile the _modified_ `mjcf_model` to a `physics` object.

# %%
# Site visualization parameters.
site_size = 3 * (0.005,)
site_rgba = (0, 1, 0, 1)

# Load MJCF model from the fly XML file.
mjcf_model = mjcf.from_path(xml_path)

# Make model modifications: add MoCap sites to legs. Don't compile model yet.

# Loop over bodies and add sites to position (0, 0, 0) in each body, which will
# correspond to leg joint locations.
for body in mjcf_model.find_all('body'):
    # If the body is a leg segment, add site to it.
    if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus_', 'claw'], body.name):
        body.add('site', name=f'mocap_{body.name}', pos=(0, 0, 0),
                 size=site_size, group=0, rgba=site_rgba)

# The mjcf_model manipulations are complete, can compile now.
physics = mjcf.Physics.from_mjcf_model(mjcf_model)

pixels = physics.render(camera_id=cameras['side'], **frame_size)
PIL.Image.fromarray(pixels)

# %% [markdown]
# # Fly model in reinforcement learning environment
# 
# So far we've considered the fly model in a "bare" stand-alone configuration. One of the main applications of the model, however, is in the context of reinforcement learning (RL) environments -- training the fly to perform different tasks, such as locomotion and other behaviors. 
# 
# `flybody` contains several ready-to-train RL environments, all of which can be created with a single line of code. Below we provide one simple example as a starting point.
# 
# The `composer` RL environment classes encapsulate the same fly model `physics` object we've seen before, and all our previous "bare fly model" examples and manipulations apply to RL environments just the same. As before, see [dm_control](https://github.com/google-deepmind/dm_control/) for more details on `composer` RL environments.

# %% [markdown]
# ## Example: fly-on-ball RL environment
# 
# This RL task models a biological experimental setup where a tethered fruit fly is required to walk on a floating ball. The image below show an actual setup for such an experiment (_image credit: Igor Siwanowicz, HHMI Janelia Research Campus_).
# 
# To emulate this experimental setup, several task-specific modifications of the fly model would be required: (i) remove the root freejoint to fuse fly's thorax with the world to imitate tethering, (ii) replace the flat floor with a ball, and (iii) add an artificial observable to allow the policy to observe the rotational state of the ball. The ball position is fixed underneath the tethered fly, but it is free to rotate if a sufficient force is applied by fly's leg. The goal of the RL task is to train a policy to control the legs such that the ball rotates in a particular direction and at a particular speed.
# 
# See [task implementation](https://github.com/TuragaLab/flybody/blob/main/flybody/tasks/walk_on_ball.py) for details on the task-specific fly model modifications, reward caclulation, and RL environment step logic.

# %% [markdown]
# ![fly-on-ball.jpg](attachment:fba018f9-7e55-4b53-a3a9-490a8b65384d.jpg)

# %% [markdown]
# The task environment can be created with the following convenient one-liner:

# %%
env = walk_on_ball()

# %% [markdown]
# We can inspect the RL environment and see the observations (policy inputs) and the actions (policy outputs). Note the extra observable `ball_qvel` we added specifically for this task. It measures the angular velocity of the ball rotation.

# %%
env.observation_spec()

# %% [markdown]
# Action specifications: shape, data type, action names, and minima and maxima of the control ranges

# %%
env.action_spec()

# %% [markdown]
# Let's reset the RL environment and visualize the initial state: the fly is stationary in its default pose, the wings are folded, and the ball is not rotating. This would be the initial state of a training episode or inference on a trained policy.

# %%
timestep = env.reset()

pixels = env.physics.render(camera_id=cameras['track1'], **frame_size)
PIL.Image.fromarray(pixels)

# %% [markdown]
# ## Run episode with random actions
# 
# Let's run a short episode with a dummy policy outputting random actions. As we run the environment loop we'll render frames and, for the sake of example, collect the reward at each time step.

# %%
n_actions = env.action_spec().shape[0]

def random_action_policy(observation):
    del observation  # Not used by dummy policy.
    random_action = np.random.uniform(-.5, .5, n_actions)
    return random_action

frames = []
rewards = []

# Environment loop.
timestep = env.reset()
for _ in range(200):
    
    action = random_action_policy(timestep.observation)
    timestep = env.step(action)
    rewards.append(timestep.reward)

    pixels = env.physics.render(camera_id=cameras['track1'], **frame_size)
    frames.append(pixels)

display_video(frames)

# %%
plt.plot(rewards)
plt.xlabel('timestep')
plt.ylabel('reward')

# %%



