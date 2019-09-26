# Reinforcement Learning for Mobile Robot Local Path Planning

The aim of this project is to train reinforcement learning agent for mobile robot control task. Project is based on ROS and Gazebo-gym environment for Turtlebot.
Agents use turtlebot lidar sensors and information from A* global path planner as input.

## Implemented agents

* [DQN](https://arxiv.org/abs/1312.5602) - RL algorithm (done)
* [Dynamic Window Approach](researchgate.net/publication/3344494_The_Dynamic_Window_Approach_to_Collision_Avoidance) - classic algorithm (done)
* [Advantage Actor Critic](papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) - RL algorithm (in progress)

You can train agnets on the Turtlebot-Lidar enviroments. Project also contains Cartpole-Environment for debugging.

### Prerequisites

Before running you have to setup ros catkin workspace with Gazebo-gym (https://github.com/erlerobot/gym-gazebo) and activate it.

Project requires python environment with:

* Pytorch-ignite
* matplotlib
* OpenAI-gym
* h5py
* pyyaml

### Run

training:
```
python train.py --env=env-maze-v0 --eps-decay=1000 --net=lstm
```

evaluatiion:
```
python eval.py --agent=dqn --env=myenv-v0 --weights out/2019_05_04_17_17_51/model_1900_total_reward_30.5
```

## License

This project is licensed under the MIT License.

## Acknowledgments

* Inspiration - https://www.mdpi.com/2076-3417/9/7/1384

