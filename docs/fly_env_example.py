# %% [markdown]
# # 果蝇强化学习环境的示例
# * 模仿飞行
# * 模拟步行
# * 视觉引导的飞行
# 
# 有关更多详细信息，请参见 `flybody` 的[出版物](https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1) 。

# %% [markdown]
# #### 要运行此脚本，请按照以下步骤操作：
# 1. 在 https://doi.org/10.25378/janelia.25309105 下载参考飞行和步行数据以及训练有素的策略
# 2. 解压下载的文件
# 3. 在下面的代码中指定数据和策略文件的路径

# %%
# wingbeat 模式发生器（pattern generator）的基线模式。
wpg_pattern_path = '../data/wing_pattern_fmech.npy'

# 飞行和步行参考数据。
ref_flight_path = '../data/flight-dataset_saccade-evasion_augmented.hdf5'
ref_walking_path = '../data/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5'

# 训练有素的策略。
flight_policy_path = '../data/policy/flight'
walk_policy_path = '../data/policy/walking'
vision_bumps_path = '../data/policy/vision-bumps'
vision_trench_path = '../data/policy/vision-trench'


# # 包导入
import numpy as np
import PIL.ImageDraw

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability
from acme import wrappers  # pip install git+https://github.com/deepmind/acme.git

from flybody.fly_envs import (
    flight_imitation,
    walk_imitation,
    vision_guided_flight,
)
from flybody.tasks.task_utils import (
    get_random_policy,
    real2canonical,
)
# from flybody.agents.utils_tf import TestPolicyWrapper
from flybody.utils import (
    display_video,
    rollout_and_render,
)

# %%
# 防止 TensorFlow 窃取所有 GPU 内存。
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# %% [markdown]
# # 有用的渲染功能

# %%
def blow(x, repeats=2):
    """Repeat columns and rows requested number of times."""
    return np.repeat(np.repeat(x, repeats, axis=0), repeats, axis=1)


def vision_rollout_and_render(env, policy, camera_id=1,
                              eye_blow_factor=5, **render_kwargs):
    """Run vision-guided flight episode and render frames, including eyes."""
    frames = []
    timestep = env.reset()
    # Run full episode until it ends.
    while timestep.step_type != 2:
        # Render eyes and scene.
        pixels = env.physics.render(camera_id=camera_id, **render_kwargs)
        eyes = eye_pixels_from_observation(
            timestep, blow_factor=eye_blow_factor)
        # Add eye pixels to scene.
        pixels[0:eyes.shape[0], 0:eyes.shape[1], :] = eyes
        frames.append(pixels)
        # Step environment.
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames


def eye_pixels_from_observation(timestep, blow_factor=4):
    """Get current eye view from timestep.observation."""
    # In the actual task, the averaging over axis=-1 is done by the visual
    # network as a pre-processing step, so effectively the visual observations
    # are gray-scale.
    left_eye = timestep.observation['walker/left_eye'].mean(axis=-1)
    right_eye = timestep.observation['walker/right_eye'].mean(axis=-1)
    pixels = np.concatenate((left_eye, right_eye), axis=1)
    pixels = np.tile(pixels[:, :, None], reps=(1, 1, 3))
    pixels = blow(pixels, blow_factor)
    half_size = pixels.shape[1] // 2
    # Add white line to separate eyes.
    pixels = np.concatenate((pixels[:, :half_size, :], 
                            255*np.ones((blow_factor*32, 2, 3)),
                            pixels[:, half_size:, :]), axis=1)
    pixels = pixels.astype('uint8')
    return pixels


def eye_pixels_from_cameras(physics, **render_kwargs):
    """Render two-eye view, assuming eye cameras have particular names."""
    for i in range(physics.model.ncam):
        name = physics.model.id2name(i, 'camera')
        if 'eye_left' in name:
            left_eye = physics.render(camera_id=i, **render_kwargs)
        if 'eye_right' in name:
            right_eye = physics.render(camera_id=i, **render_kwargs)
    pixels = np.hstack((left_eye, right_eye))
    return pixels


# 用于渲染的帧宽度和高度。
render_kwargs = {'width': 640, 'height': 480}

# %% [markdown]
# # 1. 飞行模仿环境
# 
# 在此任务中，需要果蝇模型来通过匹配其质量中心位置和身体方向来跟踪真实参考果蝇的飞行轨迹。翅膀运动由策略网络和翅膀模式发生器的组合控制。对单个策略进行了训练，可以跟踪飞行数据集中的所有轨迹。

# 让我们创建飞行模仿环境并可视化初始状态。

env = flight_imitation(ref_flight_path,
                       wpg_pattern_path,
                       terminal_com_dist=float('inf'))
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

_ = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 创建虚拟随机动作策略，运行轮次并制作视频。灰色“幽灵”果蝇指示参考飞行位置，模型需要跟踪。

random_policy = get_random_policy(env.action_spec())

frames = rollout_and_render(env, random_policy, run_until_termination=True,
                            camera_ids=1, **render_kwargs)
display_video(frames)


# %% [markdown]
# 加载训练有素的策略，运行轮次并制作视频。

flight_policy = tf.saved_model.load(flight_policy_path)
# %%
# 包装策略在测试时与非批次观察一起工作。
flight_policy = TestPolicyWrapper(flight_policy)


frames = rollout_and_render(env, flight_policy, run_until_termination=True,
                            camera_ids=1, **render_kwargs)
display_video(frames)


# %% [markdown]
# # 2. Walking imitation environment
# 
# In this task, the fly model is required to track walking trajectories of real reference flies by matching their (i) center-of-mass position, (ii) body orientation, and (iii) detailed leg movement. A single policy is trained to track all trajectories in the walking dataset.

# %% [markdown]
# Let's create walking environment and visualize the initial state.

# %%
env = walk_imitation(ref_path=ref_walking_path,
                     terminal_com_dist=float('inf'))
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

# %%
_ = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)

# %% [markdown]
# Create a dummy random-action policy, run an episode, and make video. The gray "ghost" fly indicates the reference fly position. Here, we will create a slightly different version of the random-action policy by shifting the random actions to make the video pretty.

# %%
_random_policy = get_random_policy(env.action_spec(),
                                   minimum=-.5, maximum=.5)
def random_policy(observation):
    action = _random_policy(observation)
    # Transform random action centered around zero to canonical representation
    # to match CanonicalSpecWrapper we added to the walking environment above.
    action = real2canonical(action, env._environment.action_spec())
    return action

# Request a particular (sufficiently long) walking trajectory from dataset.
env.task.set_next_trajectory_index(idx=316)

frames = rollout_and_render(env, random_policy, run_until_termination=True,
                            camera_ids=2, **render_kwargs)
display_video(frames)

# %% [markdown]
# Let's load a trained policy, run an episode, and make video.

# %%
walking_policy = tf.saved_model.load(walk_policy_path)
walking_policy = TestPolicyWrapper(walking_policy)

# %%
# Request a particular (sufficiently long) walking trajectory from dataset.
env.task.set_next_trajectory_index(idx=316)

frames = rollout_and_render(env, walking_policy, run_until_termination=True,
                            camera_ids=2, **render_kwargs)
display_video(frames)

# %% [markdown]
# # 3. Vision-guided flight over uneven terrain ("bumps")
# 
# In this task, the fly model is required to fly over an uneven sine-like terrain while maintaining a constant z-offset (height) w.r.t. the current terrain elevation. The model does not have direct access to its flight height. Instead, it has to learn to use vision to estimate the current height and to adjust it to match the current terrain elevation. Collision with terrain terminates the episode. The terrain shape is randomly re-generated in each episode.

# %% [markdown]
# Let's create "bumps" vision task environment and visualize the initial state.

# %%
env = vision_guided_flight(wpg_pattern_path, bumps_or_trench='bumps')
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

timestep = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)

# %% [markdown]
# Let's render a high-resolution view from the eye cameras in the initial episode state

# %%
pixels = eye_pixels_from_cameras(env.physics, **render_kwargs)
PIL.Image.fromarray(pixels)

# %% [markdown]
# In the same initial episode state, the actual low-resolution view available to the fly through its observables looks more like this:

# %%
pixels = eye_pixels_from_observation(timestep, blow_factor=10)
PIL.Image.fromarray(pixels)

# %% [markdown]
# Create dummy random-action policy, run episode, and make video.

# %%
random_policy = get_random_policy(env.action_spec())

frames = vision_rollout_and_render(env, random_policy, **render_kwargs)
display_video(frames)

# %% [markdown]
# Load a trained policy, run an episode, and make video:

# %%
bumps_policy = tf.saved_model.load(vision_bumps_path)
bumps_policy = TestPolicyWrapper(bumps_policy)

# %%
frames = vision_rollout_and_render(env, bumps_policy, **render_kwargs)
display_video(frames)

# %% [markdown]
# # 4. Vision-guided flight through trench
# 
# In this task, the fly is required to make it through a zigzagging trench without colliding with the trench walls. Touching the terrain or the trench walls terminates the episode. The fly has to learn to use vision to estimate its position within the trench and to maneuver to stay clear of the trench walls. The shape of the trench is randomly re-generated in each episode.

# %% [markdown]
# Let's create the "trench" task environment and visualize the initial state

# %%
env = vision_guided_flight(wpg_pattern_path, bumps_or_trench='trench')
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

# %%
_ = env.reset()
pixels = env.physics.render(camera_id=6, **render_kwargs)
PIL.Image.fromarray(pixels)

# %% [markdown]
# Let's add a new tracking camera for better trench task visualization:

# %%
# Find thorax and add tracking camera to it.
thorax = env.task.root_entity.mjcf_model.find('body', 'walker/thorax')
_ = thorax.add('camera', name='rear', mode='trackcom',
               pos=(-1.566, 0.037, -0.021),
               xyaxes=(-0.014, -1, 0, -0.012, 0, 1))

# %% [markdown]
# Visualize the initial state with the new camera:

# %%
timestep = env.reset()
trench_camera_id = env.physics.model.name2id('walker/rear', 'camera')
pixels = env.physics.render(camera_id=trench_camera_id, **render_kwargs)
PIL.Image.fromarray(pixels)

# %% [markdown]
# Let's render a high-resolution view from the eye cameras in the initial episode state:

# %%
pixels = eye_pixels_from_cameras(env.physics, **render_kwargs)
PIL.Image.fromarray(pixels)

# %% [markdown]
# And the corresponding low-resolution view used by the fly model:

# %%
pixels = eye_pixels_from_observation(timestep, blow_factor=10)
PIL.Image.fromarray(pixels)

# %% [markdown]
# As before, let's run an episode with the random-action policy:

# %%
random_policy = get_random_policy(env.action_spec())

frames = vision_rollout_and_render(
    env, random_policy, camera_id=trench_camera_id, **render_kwargs)
display_video(frames)

# %% [markdown]
# Let's load a trained policy and run an episode:

# %%
trench_policy = tf.saved_model.load(vision_trench_path)
trench_policy = TestPolicyWrapper(trench_policy)

# %%
frames = vision_rollout_and_render(
    env, trench_policy, camera_id=trench_camera_id, **render_kwargs)
display_video(frames)

# %% [markdown]
# Thank you! We are happy you are interested in our fly model:)


