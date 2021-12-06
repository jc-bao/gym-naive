import gym
from gym import spaces
import numpy as np
import logging
from gym.envs.registration import register
import matplotlib.pyplot as plt
from matplotlib import animation
import time

from numpy.core import einsumfunc

class NaiveReach(gym.GoalEnv):
    def __init__(self, config):
        self.dim = config['dim']
        self.reward_type = config['reward_type']
        self._max_episode_steps = 50
        self.err = 0.3
        self.space = spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim))
        self.action_space = spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim))
        self.observation_space = spaces.Dict(dict( # Make sure the observation dict order matchs obs!
            # observation=spaces.Box(low=np.array([-300]), high=np.array([-100])),
            observation=spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim)),
            achieved_goal=spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim)),
            desired_goal=spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim)),
        ))
        self.reset()

    def step(self, action):
        self.num_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.pos = np.clip(self.pos + action*0.5, self.space.low, self.space.high)
        obs = {
            'observation': self.pos,
            'achieved_goal': self.pos,
            'desired_goal': self.goal
        }
        d = np.linalg.norm(self.pos - self.goal)
        if self.reward_type == 'dense':
            reward = -d
        elif self.reward_type == 'sparse':
            reward = (d<self.err).astype(np.float32)
        elif self.reward_type == 'dense_diff':
            reward = self.d_old - d
            self.d_old = d
        info = {
            'is_success': (d < self.err),
        }
        done = (self.num_step >= self._max_episode_steps) or (d < self.err)
        return obs, reward, done, info

    def reset(self):
        self.num_step = 0
        self.goal = self.space.sample()
        self.pos = self.space.sample()
        self.d_old = np.linalg.norm(self.pos - self.goal)
        obs = {
            'observation': self.pos,
            'achieved_goal': self.pos,
            'desired_goal': self.goal
        }
        return obs

    def render(self):
        if self.num_step == 1:
            self.data = [self.pos]
        self.data.append(self.pos)
        if self.num_step == self._max_episode_steps or (np.linalg.norm(self.pos - self.goal))<self.err:
            if self.dim == 2:
                for i,d in enumerate(self.data):
                    plt.plot(d[0], d[1], 'o', color = [0,0,1,i/50])
                plt.plot(self.goal[0], self.goal[1], 'rx')
                plt.show()
            elif self.dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for i,d in enumerate(self.data):
                    ax.scatter(d[0], d[1], d[2], 'o', color = [0,0,1,i/50])
                plt.show()
            '''
            fig = plt.figure()
            ax = fig.add_subplot()
            point, = ax.plot([self.x[0]], [self.y[0]], 'o')
            def update_point(n, x, y, point):
                point.set_data(np.array([self.x[n], self.y[n]]))
                return point
            ani=animation.FuncAnimation(fig, func = update_point, frames = 49, interval = 1/30, fargs=(self.x, self.y, point))
            '''

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal-desired_goal)
        reward = (d<self.err).astype(np.float32)
        return reward

    def ezpolicy(self, obs):
        return obs[self.dim:] - obs[:self.dim]

register(
    id='NaiveReach-v1',
    entry_point='gym_naive.naive_reach:NaiveReach',
)

if __name__ == '__main__':
    env = NaiveReach()
    obs = env.reset()
    for i in range(50):
        act = env.ezpolicy(obs)
        obs, reward, done, info = env.step(act)
        env.render()
        print('[obs, reward, done]', obs, reward, done)
