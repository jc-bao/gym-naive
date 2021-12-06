import gym
from gym import spaces
import numpy as np

class CoinFlip(gym.Env):
    def __init__(self, config):
        self.n = config['n']
        self.reset()
        self.achieved_goal_index = 0
        self.desired_goal_index = config['n']
        self.action_space = spaces.Discrete(config['n'] + 1) # add not move
        self.observation_space = spaces.Dict(dict( # Make sure the observation dict order matchs obs!
            observation=spaces.Discrete(1),
            achieved_goal=spaces.Discrete(2**config['n']),
            desired_goal=spaces.Discrete(2**config['n']),
        ))

    def step(self, act):
        self.num_steps += 1
        if not act == self.n:
            self.coin =  self.coin ^ (1 << act)
        obs = {
            'observation': [0],
            'achieved_goal': self.coin,
            'desired_goal': self.goal
        }
        rew = float(self.coin==self.goal)
        done = rew or (self.num_steps==self.n)
        info ={
            'is_success': rew,
            'future_length': self.n - self.num_steps
        }
        return obs, rew, done, info
    
    def reset(self):
        self.coin = np.random.randint(2**self.n)
        self.goal = np.random.randint(2**self.n)
        while self.goal == self.coin:
            self.goal = np.random.randint(2**self.n)
        self.num_steps = 0
        return {
            'observation': [0],
            'achieved_goal': self.coin,
            'desired_goal': self.goal
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        return float(achieved_goal==desired_goal)
        
gym.register(
    id='CoinFlip-v0',
    entry_point='gyn_naive.coin_flip:CoinFlip',
)