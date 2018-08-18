# Asynchronous Advantage Actor-Critic for BipedalWalker-v2

The project is based on the code from this repository https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/experiments/Solve_BipedalWalker

I rewrote the code in the style being used in "tensorflow models", changed the python global variables to TensorFlow variables, added summaries for TensorBoard, added training and evaluation modes. Also I inserted additional dense layers into the Critic and Actor networks and tuned the hyperamaters in order to make the gait of the walker look more healthy.

Mostly definetely, the project will run with Gym and TensorFlow from PyPI, no need to compile TensorFlow from source.
