# bot
# load up the dependencies
import gym_super_mario_bros as gsmb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# setup the game
env = gsmb.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(SIMPLE_MOVEMENT)

done = True
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close

print("done!")