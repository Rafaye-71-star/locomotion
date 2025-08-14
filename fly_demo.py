import numpy as np
import mediapy

from flybody.fly_envs import walk_imitation

# Create walking imitation environment.
env = walk_imitation()

# Run environment loop with random actions for a bit.
for _ in range(100):
   action = np.random.normal(size=59)  # 59 is the walking action dimension.
   timestep = env.step(action)

# Generate a pretty image.
pixels = env.physics.render(camera_id=1)
# Run it in Jupyter notebook (not in Python script).
mediapy.show_image(pixels)
