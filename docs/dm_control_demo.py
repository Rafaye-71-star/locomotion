from dm_control import composer
from dm_control.locomotion.examples import basic_cmu_2019
import numpy as np

# 建立一个示例环境。
env = basic_cmu_2019.cmu_humanoid_run_walls()

# 获取描述控制输入的`action_spec`。
action_spec = env.action_spec()

# 通过随机动作逐步浏览环境。
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(action_spec.minimum, action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  print("reward = {}, discount = {}, observations = {}.".format(
      time_step.reward, time_step.discount, time_step.observation))
  

from dm_control import viewer

viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_walls)

