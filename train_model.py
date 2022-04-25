import main

# train models
state = main.env.reset()
print("state.shape: ",state.shape)
model = main.PPO('CnnPolicy', main.env, verbose=1, tensorboard_log=main.LOG_DIR, learning_rate=0.000001, n_steps=512)
model.learn(total_timesteps=10000,callback=main.callback)#add callback=callback to save models