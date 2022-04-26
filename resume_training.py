import main

# train models
state = main.env.reset()
print("state.shape: ",state.shape)
model = main.PPO.load('./saved_models/best_model_3000',tensorboard_log=main.LOG_DIR)
env = main.env
model.set_env(env)
#model = main.PPO('CnnPolicy', main.env, verbose=1, tensorboard_log=main.LOG_DIR, learning_rate=0.000001, n_steps=512)
model.learn(total_timesteps=3000,callback=main.callback, tb_log_name=main.LOG_DIR,reset_num_timesteps=False)#add callback=callback to save models