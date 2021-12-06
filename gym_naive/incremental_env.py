import gym
from gym import spaces
import numpy as np


class IncrementalEnv(gym.Env):
    def __init__(self):
        self.reset()
        self.action_space = spaces.Box(low=np.array([1]), high=np.array([5]))
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(low=np.array([0]), high=np.array([5])),
            achieved_goal=spaces.Box(low=np.array([5]), high=np.array([10])),
            desired_goal=spaces.Box(low=np.array([10]), high=np.array([15])),
        ))

    def step(self, act):
        self.num_steps += 1
        obs = {
            'observation': [self.num_steps],
            'achieved_goal':[ self.num_steps+5+act*100],
            'desired_goal': [self.num_steps+10]
        }
        rew = float(self.num_steps == np.random.randint(5,8))
        done = (self.num_steps == np.random.randint(5,10))
        info ={
            'is_success': rew,
            'future_length': 5 - self.num_steps
        }
        return obs, rew, done, info
    
    def reset(self):
        self.num_steps = 0
        return {
            'observation': [self.num_steps],
            'achieved_goal': [self.num_steps+5],
            'desired_goal': [self.num_steps+10]
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        return float(achieved_goal-desired_goal)
        
gym.register(
    id=' Incremental-v0',
    entry_point='gym_naive.naive_pac:IncrementalEnv',
)
