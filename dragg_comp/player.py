# utilities
import os
import json
import logging
from datetime import datetime
from copy import deepcopy

# redis with asyncronous implementation
from redis import StrictRedis
import redis
import pathos
import asyncio
import aioredis
import async_timeout

# openAI gym
import gym
from gym.spaces import Box

from dragg.redis_client import RedisClient
from dragg.logger import Logger
from dragg.mpc_calc import MPCCalc
from dragg_comp.agent import RandomAgent

REDIS_URL = "redis://localhost"

class PlayerHome(gym.Env):
    def __init__(self):
        self.nstep = 0
        asyncio.run(self.await_status("ready"))
        home = self.set_home()
        self.home = MPCCalc(home)
        self.name = self.home.name
        with open('data/rl_data/state_action.json','r') as file:
            states_actions = json.load(file)
        self.states = [k for k, v in states_actions['states'].items() if v]
        self.observation_space = Box(-1, 1, shape=(len(self.states), ))
        self.actions = [k for k, v in states_actions['actions'].items() if v]
        self.action_space = Box(-1, 1, shape=(len(self.actions), ))
        asyncio.run(self.post_status("initialized player"))
        asyncio.run(self.await_status("all ready"))

    def reset(self):
        """
        Reset as required by OpenAI gym. Beta implementation simply returns current observation, 
        meaning that the simulation will overall continue running. 
        :return: state vector of length n
        """
        obs = self.get_obs()

        return obs 

    def set_home(self):
        """
        Gets the first home in the queue (broadcast by the Aggregator).
        :return: MPCCalc object
        :input: None
        """
        redis_client = RedisClient()
        self.num_timesteps = int(redis_client.conn.hgetall("simulation")['nsteps'])
        home = redis_client.conn.hgetall("home_values")
        home['hvac'] = redis_client.conn.hgetall("hvac_values")
        home['wh'] = redis_client.conn.hgetall("wh_values")
        home['hems'] = redis_client.conn.hgetall("hems_values")
        home['hems']["weekday_occ_schedule"] = [[19,8],[17,18]]
        if 'battery' in home['type']:
            home['battery'] = redis_client.conn.hgetall("battery_values")
        if 'pv' in home['type']:
            home['pv'] = redis_client.conn.hgetall("pv_values")
        home['wh']['draw_sizes'] = [float(i) for i in redis_client.conn.lrange('draw_sizes', 0, -1)]
        home['hems']['weekday_occ_schedule'] = redis_client.conn.lrange('weekday_occ_schedule', 0, -1)
        print(f"Welcome {home['name']}")

        return home

    def get_obs(self):
        """
        Gets the corresponding values for each of the desired state values, as set in state_action.json.
        User can change this method according to how it post processes any observation values and/or in what values it receives.
        :return: list of float values
        """
        obs = []
        for state in self.states:
            if state in self.home.optimal_vals.keys():
                obs += [self.home.optimal_vals[state]]
            else:
                obs += [0]

        return obs

    def get_reward(self):
        """ 
        Determines a reward, function can be redefined by user in any way they would like.
        :return: float value normalized to [-1,1] 
        """
        reward = self.redis_client.conn.hget("current_values", "current_demand")
        return reward

    def step(self, action=None):
        """
        :input: action (list of floats)
        Redefines the OpenAI Gym environment step.
        :return: observation (list of floats), reward (float), is_done (bool), debug_info (set)
        """
        print("stepping forward")
        self.nstep += 1
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        fh.setLevel(logging.WARN)

        self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        self.redis_client = RedisClient()
        self.home.redis_get_initial_values()
        self.home.cast_redis_timestep()

        if self.home.timestep > 0:
            self.home.redis_get_prev_optimal_vals()

        if action:
            temp = self.home.t_in_max 
            start = (self.home.timestep * self.home.sub_subhourly_steps * self.home.dt) % (24 * self.home.sub_subhourly_steps * self.home.dt)
            stop = start + (self.home.dt * self.home.sub_subhourly_steps)
            self.home.t_in_max[start:stop] = [action + 0.5 * self.home.t_deadband] * self.home.dt * self.home.sub_subhourly_steps
            self.home.t_in_min[start:stop] = [action - 0.5 * self.home.t_deadband] * self.home.dt * self.home.sub_subhourly_steps

        self.home.get_initial_conditions()
        self.home.solve_type_problem()
        self.home.cleanup_and_finish()
        self.home.redis_write_optimal_vals()

        self.home.log.removeHandler(fh)

        if action:
            print(self.home.timestep)
            self.home.t_in_max = temp # reset to default values (@akp, clean up)

        asyncio.run(self.post_status("updated"))
        asyncio.run(self.await_status("forward"))

        return self.get_obs(), self.get_reward(), False, {}

    async def await_status(self, status):
        """
        :input: Status (string)
        Opens and asynchronous reader and awaits the specified status
        :return: None
        """
        async_redis = aioredis.from_url(REDIS_URL)
        pubsub = async_redis.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")

        i = 0
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await pubsub.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        print(f"(Reader) Message Received: {message}")
                        if status in message["data"].decode():
                            break
                        else:
                            await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                pass
        return

    async def post_status(self, status):
        """
        :input: Status (string)
        Publishes a status (typically "is done" to alert the aggregator)
        :return: None
        """
        async_redis = aioredis.from_url(REDIS_URL)
        pubsub = async_redis.pubsub()
        await pubsub.subscribe("channel:1")
        print(f"{self.home.name} {status} at t = {self.nstep}.")
        await async_redis.publish("channel:1", f"{self.home.name} {status} at t = {self.nstep}.")
        return 

if __name__=="__main__":
    tic = datetime.now()
    my_home = PlayerHome()
    agent = RandomAgent(my_home)

    for _ in range(my_home.num_timesteps * my_home.home.dt):
        action = my_home.action_space.sample()
        my_home.step() 

    asyncio.run(my_home.post_status("done"))
    toc = datetime.now()
    print(toc-tic)
