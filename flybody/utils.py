"""实用功能。"""

from typing import Sequence

import numpy
# from IPython.display import HTML
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def rollout_and_render(env, policy, n_steps=100,
                       run_until_termination=False,
                       camera_ids=[-1],
                       **render_kwargs):
    """Rollout policy for n_steps or until termination, and render video.
    Rendering is possible from multiple cameras; in that case, each element in
    returned `frames` is a list of cameras."""
    # 放在函数外会在 github action 中没有 GPU，导致 import tf 失败
    from acme.tf import utils as tf2_utils
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]
    timestep = env.reset()
    frames = []
    i = 0
    while ((i < n_steps and not run_until_termination) or 
           (timestep.step_type != 2 and run_until_termination)):
        i += 1
        frame = []
        for camera_id in camera_ids:
            frame.append(
                env.physics.render(camera_id=camera_id, **render_kwargs))
        frame = frame[0] if len(camera_ids) == 1 else frame  # Maybe squeeze.
        frames.append(frame)
        # 没有使用分布式策略，在这里补充TestPolicyWrapper里的一些操作
        batched_observation = tf2_utils.add_batch_dim(timestep.observation)
        distribution = policy(batched_observation)
        action = distribution.mean()  # 不是测试模式（测试模式返回均值和方差）
        if type(action) is not numpy.float64:
            action = action[0, :].numpy()  # Remove batch dimension.
        # action = policy(timestep.observation)
        timestep = env.step(action)
    return frames


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)


def display_video(frames, framerate=30):
    """
    Args:
        frames (array): (n_frames, height, width, 3)
        framerate (int)
    """
    height, width, _ = frames[0].shape
    dpi = 70
    # orig_backend = matplotlib.get_backend()
    # matplotlib.use(
    #     'TkAgg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    # plt.close(
    #    'all')  # Figure auto-closing upon backend switching is deprecated.
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig,
                                   func=update,
                                   frames=frames,
                                   interval=interval,
                                   blit=True,
                                   repeat=False)
    plt.show()
    plt.close()  # 需要手动关闭图形窗口才能运行下一个图
    print(anim)
    # return HTML(anim.to_html5_video())


def parse_mujoco_camera(s: str):
    """Parse `Copy camera` XML string from MuJoCo viewer to pos and xyaxes.
    
    Example input string: 
    <camera pos="-4.552 0.024 3.400" xyaxes="0.010 -1.000 0.000 0.382 0.004 0.924"/>
    """
    split = s.split('"')
    pos = split[1]
    xyaxes = split[3]
    pos = [float(s) for s in pos.split()]
    xyaxes = [float(s) for s in xyaxes.split()]
    return pos, xyaxes
