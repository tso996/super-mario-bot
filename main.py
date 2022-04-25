# bot
# load up the dependencies
from time import sleep
import gym_super_mario_bros as gsmb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback #saving models



# ===============================================
# each model is around 300MB, so be careful how often the models are saved.
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
# ===============================================

# setup the game and paths
CHECKPOINT_DIR =  './saved_models/'
LOG_DIR = './logs/'
#saves models every 10k steps
callback = TrainAndLoggingCallback(check_freq=1000,save_path=CHECKPOINT_DIR)
#AI model initialised, CnnPolicy(convolutional neural networks) is very efficient at image processing

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
# plt.imshow(state[0])
# plt.show()
# # sleep(10)
# print(SIMPLE_MOVEMENT)
# print(env.step(env.action_space.sample()))
# exit()
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=1024)
model.learn(total_timesteps=1025,callback=callback)#add callback=callback to save models
exit()

done = True
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close



print("done!")