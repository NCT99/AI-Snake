# AI-Snake
Reinforcement learning approach to the snake game.
# Summary
This code is a simple implementation of the q-learning method of reinforcement learning applied to snake. The agent learns to estimate the value of an action in a specific state by repeated trial and error combined with a neural network. After this training phase we get an agent that chooses in every step the action it considers optimal, this is the final version found here.
# Prerequisites
To run the project you will need Python 3.6, pygame, Tensorflow and Keras.
# Usage
To use simply download the folder and see the trained AI in action by running snake.py. The algorithm will use the weights found in the weights folder which were fixed after around an hour of training, after that I modified some parts of the code so that the AI would remain fixed. If you wish to train it for longer (which would definitely improve it) you can do it with some very simple changes to the code. 
