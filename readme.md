-This project uses the OpenAI Gym. https://gym.openai.com/

![](demo.gif)

-It also uses the nes-py to use the controller to play the game.

-Install the dependencies from the requirements.txt.

```pip3 install -r requirements.txt```

-Do 2 types of preprocessing before applying reinforcement learning

    -converting the rgb frames to gray

    -frame stacking

-Use the PPO algorithm

-To view the tensorboard statistics run from logs/PPO_20

```tensorboard --logdir=.```