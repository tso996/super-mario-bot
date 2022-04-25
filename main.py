# bot
# load up the dependencies
from time import sleep
import gym_super_mario_bros as gsmb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt


# setup the game
env = gsmb.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')#last few frames will be in the stack to use

state = env.reset()
print("state.shape: ",state.shape)
#==================
#this code shows the stack 
# for step in range(5):
#     state, reward, done, info = env.step([5])
# print("state.shape: ",state.shape)
# plt.figure(figsize=(10,8))
# for idx in range(state.shape[3]):
#     plt.subplot(1,4,idx+1)
#     plt.imshow(state[0][:,:,idx])
# # plt.imshow(state[0])
# plt.show()
#================
plt.imshow(state[0])
plt.show()
# sleep(10)
exit()

print(SIMPLE_MOVEMENT)
# print(env.step(env.action_space.sample()))
# exit()

done = True
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close

print("done!")