# utilities
import os
import json
import logging
from datetime import datetime
from copy import deepcopy
import numpy as np
import pandas as pd
import pathos

# redis with asyncronous implementation
from redis import StrictRedis
import redis
import asyncio
import aioredis
import async_timeout
import selectors

# openAI gym
import gym
from gym.spaces import Box

import dragg.redis_client as rc
from dragg.logger import Logger
from dragg.mpc_calc import MPCCalc
from dragg_comp.agent import RandomAgent


REDIS_URL = "redis://localhost"

class PlayerHome(gym.Env):
    def __init__(self, redis_url=REDIS_URL, home_dict=None):
        """
        PlayerHome should be initialized with a REDIS_URL. Initialization with home_dict
        as an explicit input should be reserved for unit testing of player submissions.
        """
        self.log = Logger("player")
        self.nstep = 0
        self.redis_url = redis_url
        asyncio.run(self.await_status("ready"))
        if redis_url:
            home_dict = self.get_home_redis()
        self.home = MPCCalc(home_dict)
        self.name = self.home.name
        with open('data/state_action.json','r') as file:
            states_actions = json.load(file)
        self.states = [k for k, v in states_actions['states'].items() if v]
        self.observation_space = Box(-1, 1, shape=(len(self.states), ))
        self.actions = [k for k, v in states_actions['actions'].items() if v]
        self.action_space = Box(-1*np.ones(len(self.actions)), np.ones(len(self.actions)))
        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(selector)
        asyncio.set_event_loop(loop)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.post_status("initialized player"))
        loop.run_until_complete(self.await_status("all ready"))
        loop.close()
        self.demand_profile = []
        self.reset(initialize=True)

    def update_states(self, obs_dict):
        """
        Resets the state space to accomodate custom filtering by players. (Defaults
        to a state space that utilizes all available data.)
        :return: None
        """
        self.states = [k for k, v in obs_dict.items() if v]
        self.observation_space = Box(-1, 1, shape=(len(self.states), ))
        return

    def reset(self, initialize=False):
        """
        Reset as required by OpenAI gym. Beta implementation simply returns current observation, 
        meaning that the simulation will overall continue running. 
        :return: state vector of length n
        """
        self.log.logger.info("Resetting the player's environment.")
        asyncio.run(self.post_status("reset"))
        self.nstep = 0

        if initialize:
            self.home.redis_get_initial_values()
            self.home.cast_redis_timestep()
            self.home.get_initial_conditions()
            self.home.add_base_constraints()
            self.home.set_p_grid()
            self.home.solve_mpc(debug=True)
            self.home.cleanup_and_finish()

        obs = self.get_obs()

        return obs 

    def get_home_redis(self):
        """
        Gets the first home in the queue (broadcast by the Aggregator).
        :return: MPCCalc object
        :input: None
        """
        redis_client = rc.connection(self.redis_url)#RedisClient()
        self.num_timesteps = int(redis_client.hgetall("simulation")['nsteps'])
        home = redis_client.hgetall("home_values")
        home['hvac'] = redis_client.hgetall("hvac_values")
        home['wh'] = redis_client.hgetall("wh_values")
        home['hems'] = redis_client.hgetall("hems_values")
        home['hems']['horizon'] = 1
        if 'battery' in home['type']:
            home['battery'] = redis_client.hgetall("battery_values")
        if 'pv' in home['type']:
            home['pv'] = redis_client.hgetall("pv_values")
        home['wh']['draw_sizes'] = [float(i) for i in redis_client.lrange('draw_sizes', 0, -1)]
        home['hems']['weekday_occ_schedule'] = redis_client.lrange('weekday_occ_schedule', 0, -1)
        self.log.logger.info(f"Welcome {home['name']}")
        return home

    def get_obs(self):
        """
        Gets the corresponding values for each of the desired state values, as set in state_action.json.
        User can change this method according to how it post processes any observation values and/or in what values it receives.
        :return: list of float values
        """
        obs = []
        self.obs_dict = {}
        for state in self.states:
            skip = False
            if state in self.home.optimal_vals.keys():
                obs += [self.home.optimal_vals[state]]
            # elif state == "leaving_horizon":
                # obs += [self.home.ev.index_8am[0] if self.home.ev.index_8am else -1]
            # elif state == "returning_horizon":
                # obs += [self.home.ev.index_5pm[0] if self.home.ev.index_5pm else -1]
            elif state == "occupancy_status":
                obs += [int(self.home.occ_on[0])]
            elif state == "future_waterdraws":
                obs += [np.sum(self.home.optimal_vals["waterdraws"])]
            elif state == "t_out":
                obs += [self.home.all_oat[self.home.start_slice]]
            elif state == "t_out_6hr":
                obs += [self.home.all_oat[self.home.start_slice + 6*self.home.dt]]
            elif state == "t_out_12hr":
                obs += [self.home.all_oat[self.home.start_slice + 12*self.home.dt]]
            elif state == "ghi":
                obs += [self.home.all_ghi[self.home.start_slice]]
            elif state == "ghi_6hr":
                obs += [self.home.all_ghi[self.home.start_slice + 6*self.home.dt]]
            elif state == "ghi_12hr":
                obs += [self.home.all_ghi[self.home.start_slice + 12*self.home.dt]]
            elif state == "t_in":
                obs += [self.home.optimal_vals["temp_in_opt"]]
            elif state == "t_wh":
                obs += [self.home.optimal_vals["temp_wh_opt"]]
            elif state == "e_ev":
                obs += [self.home.optimal_vals["e_ev_opt"]]
            elif state == "time_of_day":
                tod = self.home.timestep % (24 * self.home.dt)
                obs += [tod]
            elif state == "community_demand":
                community_demand = self.home.redis_client.hget("current_values", "current_demand")
                if not community_demand:
                    community_demand = 0
                obs += [community_demand / (self.home.max_load / 5) - 1]
            elif state == "my_demand":
                obs += [2 * self.home.optimal_vals["p_grid_opt"] / self.home.max_load - 1]
            elif state == "day_of_week":
                obs += [self.home.weekday_current[0]]
            else:
                skip = True
                self.log.logger.warn(f"MISSING {state}")

            if not skip:
                self.obs_dict.update({state:obs[-1]})

        return obs

    def get_reward(self):
        """ 
        Determines a reward, function can be redefined by user in any way they would like.
        :return: float value normalized to [-1,1] 
        """
        reward = 0
        return reward

    def score(self):
        """
        Calculates a score for the player in the game.
        :return: dictionary of key performance indexes
        """
        redis_client = rc.connection(self.redis_url)
        contribution2peak = float(redis_client.hget("peak_contribution", self.name))
        kpis = { 
            "l2_norm": [np.linalg.norm(self.demand_profile)], 
            "contribution2peak": [contribution2peak]
            }

        kpis_df = pd.DataFrame(kpis)
        kpis_df.to_csv("outputs/score.csv")

        return kpis

    def step(self, action=None):
        """
        :input: action (list of floats)
        Redefines the OpenAI Gym environment step.
        :return: observation (list of floats), reward (float), is_done (bool), debug_info (set)
        """
        action = list(action)
        self.nstep += 1
        if not os.path.isdir("home_logs"):
            os.mkdir("home_logs")
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        fh.setLevel(logging.WARN)

        self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        self.redis_client = rc.connection(self.redis_url)
        self.home.redis_get_initial_values()
        self.home.cast_redis_timestep()

        if self.home.timestep > 0:
            self.home.redis_get_prev_optimal_vals()

        self.home.get_initial_conditions()
        self.home.add_base_constraints()
        if action is not None:
            if "ev_charge" in self.actions and "ev" in self.home.devices:
                self.home.constraints += self.home.ev.override_charge(action[2]) # overrides the p_ch for the electric vehicle
            if "wh_setpoint" in self.actions and "wh" in self.home.devices:
                self.home.constraints += self.home.wh.override_p_wh(action[1]) # same but for waterheater
            if "hvac_setpoint" in self.actions and "hvac" in self.home.devices:
                self.home.constraints += self.home.hvac.override_t_in(action[0]) # changes thermal deadband to new lower/upper bound
        self.home.set_p_grid()
        # self.home.solve_mpc(debug=True)
        self.home.solve_local_control()
        self.home.cleanup_and_finish()
        self.home.redis_write_optimal_vals()

        self.home.log.removeHandler(fh)

        asyncio.run(self.post_status("updated"))
        asyncio.run(self.await_status("forward"))

        self.demand_profile += [self.home.optimal_vals["p_grid_opt"]]

        return self.get_obs() #, self.get_reward(), False, {}

    async def await_status(self, status):
        """
        :input: Status (string)
        Opens and asynchronous reader and awaits the specified status
        :return: None
        """
        async_redis = aioredis.from_url(self.redis_url)
        pubsub = async_redis.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")

        i = 0
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await pubsub.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        self.log.logger.debug(f"(Reader) Message Received: {message['data'].decode()}")
                        if status in message["data"].decode():
                            break
                        else:
                            await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                print("player timed out")
                pass
        return

    async def post_status(self, status):
        """
        :input: Status (string)
        Publishes a status (typically "is done" to alert the aggregator)
        :return: None
        """
        async_redis = aioredis.from_url(self.redis_url)
        pubsub = async_redis.pubsub()
        await pubsub.subscribe("channel:1")
        self.log.logger.info(f"{self.home.name} {status} at t = {self.nstep}.")
        await async_redis.publish("channel:1", f"{self.home.name} {status} at t = {self.nstep}.")
        return 

if __name__=="__main__":
    import random 
    tic = datetime.now()
    my_home = PlayerHome()

    for _ in range(my_home.num_timesteps):
        action = my_home.action_space.sample()
        my_home.step(action) 

    asyncio.run(my_home.post_status("done"))
    print(my_home.score())
    toc = datetime.now()
    print(toc-tic)
