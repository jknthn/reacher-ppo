# Reacher PPO

This is a code for 2nd project in Udacity Deep Reinforcement Learning Nanodegree. I chose Proximal Policy Optimization algorithm as it was both faster in terms of computing power as well as it took less episode to converge than alternatives (eg. Deep Deterministic Policy Gradient).

![](images/train_end.gif)

## Environment

Agent was learned on Reacher environment with `20` simultanous agents for faster rollout gathering. Environment is solved when average reward over all agents in last 100 episodes is over `30.0`. 

```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		goal_speed -> 1.0
		goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

## Getting started

Make sure you have python 3.6 installed and virtual environment of your choosing activated. Unity has to be installed on your system. Run:

```source ./install.sh```

to install python dependencies. Then you should be able to run jupyter notebook and view `Solution.ipynb`. File `model.py` contains neural network class used as a policy estimator and `agent.py` contains logic for the PPO agent. You can find trained model weights in a `/models` directory.

## Instructions

Run `Soultion.ipynb` for further details.

## Sources

[1] Original PPO paper by Open AI [Proximal Policy Optimization Algorithm](https://arxiv.org/pdf/1707.06347.pdf)
[2] Repository of PyTorch Deep RL implementations [DeepRL](https://github.com/ShangtongZhang/DeepRL)