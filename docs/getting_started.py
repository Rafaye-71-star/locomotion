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
# 现在我们可以模拟运动序列。在每个控制步骤中，我们将将控件写入Mujoco的`mjdata->ctrl`中，以`physics.data.ctrl`暴露出来。我们还将在运动序列期间的某个点接合腿粘附执行器。
# 
# 控制语义是位置执行器的目标关节角，而（缩放）力和粘附执行器的力（请参见[Mujoco 文档](https://mujoco.readthedocs.io/en/stable/stable/overview.html) ）。
# 除翅膀和粘附外，我们的模型使用位置执行器。
# 翅膀由力（扭矩）执行器提供动力。
# 
# 我们将每次更改控件一次`physics_to_ctrl_ratio == 10`仿真步骤，以确保仿真稳定性（更多详细信息请参阅`dm_control`的 [论文](https://arxiv.org/abs/2006.12983) 。
# 
# 请注意，在运动序列的第一部分“让翅膀折叠”中，我们没有更改`physics.data.ctrl`这一变量，并且在用`physics.reset()`重置模拟后，该变量的值仍然为零。
# 然而，如上所述，翅膀会折叠 -- 这是由弱弹簧作用将翅膀移至参考位置的弱弹簧来实现的。

n_steps = 100
physics_to_ctrl_ratio = 10
frames = []

# 将物理重置为初始默认状态。
physics.reset()

# 让翅膀折叠。
for i in range(n_steps):
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()

# 扭头。
for i in range(n_steps):
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)
    physics.named.data.ctrl['head_twist'] = 1.5 * np.sin(2*np.pi * i/n_steps)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()

# 移动中间的右侧腿。
for i in range(n_steps+50):
    if i <= n_steps:
        physics.named.data.ctrl[leg_actuators_L2] = amplitude_mid * np.sin(np.pi * i/n_steps)
    for _ in range(physics_to_ctrl_ratio):
        physics.step()
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    frames.append(pixels)

# 激活中间腿粘附以防止稍后提起前腿时滑倒。
physics.named.data.ctrl[['adhere_claw_T2_right', 'adhere_claw_T2_left']] = 1.

# 随后抬起前腿。
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
# ## 通过随机动作驱动整个身体
# 现在，让我们通过随机控制立即启动所有自由度。
# 
# 由于力执行器（翅膀）和位置执行器（身体其余部分）具有不同的控制语义（分别为缩放力和目标关节角），我们将使用不同幅度的控制信号来执行它们。让我们首先找到每个执行器组的索引：

wing_act_indices = []  # 力执行器。
body_act_indices = []  # 位置执行器。
# 在所有执行器上循环。
for i in range(physics.model.nu):
    name = physics.model.id2name(i, 'actuator')
    # 分别存储翅膀执行器索引和其余索引。
    if 'wing' in name:
        wing_act_indices.append(i)
    else:
        body_act_indices.append(i)

print(wing_act_indices)
print(body_act_indices)


# %% [markdown]
# 运行模拟并通过随机动作来启动果蝇。

n_body_actions = len(body_act_indices)
n_wing_actions = len(wing_act_indices)

n_steps = 300
physics_to_ctrl_ratio = 10
frames = []

# 将模拟重置为初始默认状态。
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
# ## 模型修改：使用[PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/main/main/dm_control/mjcf/mjcf/mheadme.md) 添加腿部 MoCap 位点
# 
# 除了操纵关节角度和控件外，就像我们上面所做的那样，可以修改果蝇模型本身。模型的所有方面（例如，身体部位的大小、执行器强度和类型、自由度、质量、外观等）可以通过编程方式轻松更改。
# 
# 让我们考虑一个简单的苍蝇身体修改的例子。假设我们有一个运动捕获数据集在跟踪果蝇腿关节的位置，我们想拟合果蝇模型的腿部姿势。这样做的一种方法将需要引用果蝇腿中相应的关键点位置，这又可以通过将位点添加到腿关节中来实现。
# 
# 与以前一样，将模型加载和编译到物理`physics`对象相比，我们将把这个过程分为两个步骤。
# 首先，我们将加载果蝇模型为`mjcf_model`，这是基础 Mujoco Fly XML 文件的 Python 对象模型，我们可以通过编程方式进行互动并修改（请参阅`dm_control` 的 [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md) ）。
# 其次，我们将 _修改的_ `mjcf_model` 编译到物理`physics`对象。

# 位点可视化参数。
site_size = 3 * (0.005,)
site_rgba = (0, 1, 0, 1)

# 从果蝇 XML 文件加载 MJCF 模型。
mjcf_model = mjcf.from_path(xml_path)

# 进行模型修改：将 MoCap 位点添加到腿部。不要编译模型。

# 在身体上循环，并将位点添加到每个身体中的位置 (0, 0, 0)，这将对应于腿关节位置。
for body in mjcf_model.find_all('body'):
    # If the body is a leg segment, add site to it.
    # 如果身体是腿部的片段，请在其上添加位点。
    if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus_', 'claw'], body.name):
        body.add('site', name=f'mocap_{body.name}', pos=(0, 0, 0),
                 size=site_size, group=0, rgba=site_rgba)

# mjcf_model 操作已经完成，现在可以编译。
physics = mjcf.Physics.from_mjcf_model(mjcf_model)

pixels = physics.render(camera_id=cameras['side'], **frame_size)
PIL.Image.fromarray(pixels)


# %% [markdown]
# # 强化学习环境中的果蝇模型
# 
# 到目前为止，我们一直是在“纯”独立配置的环境下对果蝇模型进行了研究。
# 然而，该模型的主要应用之一是在强化学习（RL）环境当中训练果蝇来完成不同的任务，比如移动和其它行为。
# 
# `flybody`包含几个现成的RL环境，所有这些环境都可以使用一行代码创建。在下面，我们提供一个简单的示例作为起点。
# 
# `composer` RL 环境类封装了我们之前见过的相同的飞行模型`physics`对象和所有之前的“裸果蝇模型”示例，并且操作适用于RL环境。如前所述，有关`composer` 强化学习环境的更多详细信息，请参见 [dm_control](https://github.com/google-deepmind/dm_control/) 。

# ## 示例：在球上果蝇的强化学习环境
# 
# 这项强化学习任务对生物学实验设置进行了建模，其中需要束缚果蝇才能在浮动的球上行走。下图显示了此类实验的实际设置（_图片来源：Igor Siwanowicz，HHMI Janelia Research Campus_）。
# 
# 为了模仿这种实验设置，需要对飞行模型进行几种特定于任务的修改：
# （i）移除根部自由关节，将果蝇的胸部与外部环境融合在一起，以实现类似束缚的效果，
# （ii）用球体代替平坦的地板，
# （iii）添加可观察的人造观察者，以允许策略观察球的旋转状态。
# 球位置固定在束缚的苍蝇下方，但是如果果蝇的腿施加了足够的力，则可以自由旋转。
# 强化学习任务的目的是训练控制腿部的策略，以使球以特定的方向和特定速度旋转。
# 
# 有关任务特定的果蝇模型修改、奖励计算以及强化学习环境的步骤逻辑等详细信息，请参阅 [任务实施](https://github.com/TuragaLab/flybody/blob/main/flybody/tasks/walk_on_ball.py) 部分。

# 可以使用以下方便的单行小程序来创建任务环境：
env = walk_on_ball()

# 我们可以检查强化学习环境，并查看观测值（策略输入）和动作（策略输出）。请注意，我们专门为此任务添加了额外可观察的`ball_qvel`。它测量球旋转的角速度。
env.observation_spec()

# 动作规格：控制范围的形状、数据类型、动作名称以及最小和最大值
env.action_spec()


# 让我们重置强化学习环境并将其可视化初始状态：果蝇在其默认姿势中静止不动，翅膀折叠，球不旋转。这将是训练轮次的初始状态或对训练有素策略的推断。
timestep = env.reset()

pixels = env.physics.render(camera_id=cameras['track1'], **frame_size)
PIL.Image.fromarray(pixels)


# %% [markdown]
# ## 使用随机动作运行轮次
# 
# 让我们以一个虚拟策略输出随机动作来完成简短的轮次。在运行环境循环时，我们将渲染帧，并为了举例来说，在每个时间步骤中收集奖励。
n_actions = env.action_spec().shape[0]

def random_action_policy(observation):
    del observation  # 虚拟策略不使用。
    random_action = np.random.uniform(-.5, .5, n_actions)
    return random_action

frames = []
rewards = []

# 环境循环。
timestep = env.reset()
for _ in range(200):
    action = random_action_policy(timestep.observation)
    timestep = env.step(action)
    rewards.append(timestep.reward)

    pixels = env.physics.render(camera_id=cameras['track1'], **frame_size)
    frames.append(pixels)

display_video(frames)

plt.plot(rewards)
plt.xlabel('timestep')
plt.ylabel('reward')


