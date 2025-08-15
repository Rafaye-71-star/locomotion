import numpy as np
import mediapy

from flybody.fly_envs import walk_imitation

# 创建一个模仿行走的环境。
env = walk_imitation()

# 使用随机操作进行环境循环。
for _ in range(100):
   action = np.random.normal(size=59)  # 59是步行动作维度。
   timestep = env.step(action)

# 生成漂亮的图像。
pixels = env.physics.render(camera_id=1)
# 在 Jupyter notebook 中运行它（不是在Python脚本中）。
# vscode: right click -> Run in Interactive Window -> Run Current File in Interactive Window
mediapy.show_image(pixels)
