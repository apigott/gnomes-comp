from datetime import datetime
import asyncio
import argparse

from dragg_comp.player import PlayerHome

REDIS_URL = "redis://localhost"

class RLTrainingEnv(PlayerHome):
	def __init__(self, redis_url=REDIS_URL, normalization, reward):
		"""
		Creates a custom implementation of the GNOMES environment based on the OpenAI gym environment.
		Players should pass a normalization function (to return a list of states, given the env) and
		a reward function (to return a reward given the env)
		"""
		self.normalization = normalization
		self.reward = reward
		super().__init__(redis_url=redis_url)
		
	def get_reward(self):
		"""
		Redefines get_reward with the player's implementation
		"""
		return self.reward(self)

	def reset(self, initialize=False):
		"""
		Resets the environment to the initial start datetime and env conditions
		"""
		super().reset()
		return self.normalization(self)

	def step(self, action):
		"""
		Steps the environment forward.
		:input: action (vector of length 3)
		:output: (state, reward, done, info)
		"""
		obs = super().step(action)
		reward = self.get_reward()
		return self.normalization(self), reward, False, {}