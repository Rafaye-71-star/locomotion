import os
# 不注释在Windows下报错： ImportError: ('Unable to load EGL library', "Could not find module 'EGL' (or one of its dependencies). Try using the full path with constructor syntax.", 'EGL', None)
# os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

print(torch.cuda.is_available())


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


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	评估单任务/多任务 TD-MPC2 检查点的脚本

	最相关的参数：
		`task`: 任务名 (或者对于多任务评估选择 mt30/mt80 )
		`checkpoint`: 加载模型检查点的路径
		`eval_episodes`: 在每个任务上评估的轮次数目 (默认: 10)
		`save_video`: 是否保存评估的视频 (默认：True)
		`seed`: 随机种子 (默认：1)
	
	完整的参数列表请查看 config.yaml

	示例使用：
	````
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	# 创建环境
	env = make_env(cfg)

	# 加载智能体
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# 评估
	print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	print(f'Evaluation episodes: {cfg.eval_episodes}')
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	ep_rewards, ep_successes = [], []
	for i in tqdm(range(cfg.eval_episodes), desc=f'{cfg.task}'):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if cfg.save_video:
			frames = [env.render()]
		while not done:
			action = agent.act(obs, t0=t==0)
			obs, reward, done, info = env.step(action)
			ep_reward += reward
			t += 1
			if cfg.save_video:
				frames.append(env.render())
		ep_rewards.append(ep_reward)
		ep_successes.append(info['success'])
		if cfg.save_video:
			frames = np.stack(frames)
			imageio.mimsave(
				os.path.join(video_dir, f'{cfg.task}-{i}.mp4'), frames, fps=15)
	ep_rewards = np.mean(ep_rewards)
	ep_successes = np.mean(ep_successes)
	print(colored(f'  {cfg.task:<22}' \
		f'\tR: {ep_rewards:.01f}  ' \
		f'\tS: {ep_successes:.02f}', 'yellow'))
	

if __name__ == '__main__':
	evaluate()
