# utilities
import os
import json
import logging
from datetime import datetime
from copy import deepcopy
import numpy as np

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

# from dragg.redis_client import RedisClient
import dragg.redis_client as rc
from dragg.logger import Logger
from dragg.mpc_calc import MPCCalc
from dragg_comp.agent import RandomAgent

REDIS_URL = "redis://localhost"

class PlayerHome(gym.Env):
    def __init__(self, redis_url=REDIS_URL):
        self.nstep = 0
        self.redis_url = redis_url
        asyncio.run(self.await_status("ready"))
        home = self.set_home()
        self.home = MPCCalc(home)
        self.name = self.home.name
        with open('data/state_action.json','r') as file:
            states_actions = json.load(file)
        self.states = [k for k, v in states_actions['states'].items() if v]
        self.observation_space = Box(-1, 1, shape=(len(self.states), ))
        self.actions = [k for k, v in states_actions['actions'].items() if v]
        a_min = []
        a_max = []
        for action in self.actions:
            if action == "hvac_setpoint":
                a_min += [16]
                a_max += [24]
            elif action == "wh_setpoint":
                a_min += [42]
                a_max += [52]
            elif action == "ev_charge":
                a_min += [-1]
                a_max += [1]
        self.action_space = Box(np.array(a_min), np.array(a_max))
        asyncio.run(self.post_status("initialized player"))
        asyncio.run(self.await_status("all ready"))
        self.demand_profile = []
        self.reset(initialize=True)

    def reset(self, initialize=False):
        """
        Reset as required by OpenAI gym. Beta implementation simply returns current observation, 
        meaning that the simulation will overall continue running. 
        :return: state vector of length n
        """
        if initialize:
            self.home.redis_get_initial_values()
            self.home.cast_redis_timestep()
            self.home.get_initial_conditions()
            self.home.add_type_constraints()
            self.home.set_type_p_grid()
            self.home.solve_mpc(debug=True)
            self.home.cleanup_and_finish()

        obs = self.get_obs()

        return obs 

    def set_home(self):
        """
        Gets the first home in the queue (broadcast by the Aggregator).
        :return: MPCCalc object
        :input: None
        """
        redis_client = rc.connection(self.redis_url)#RedisClient()
        print("!!!", redis_client.hgetall("simulation"))
        self.num_timesteps = int(redis_client.hgetall("simulation")['nsteps'])
        home = redis_client.hgetall("home_values")
        home['hvac'] = redis_client.hgetall("hvac_values")
        home['wh'] = redis_client.hgetall("wh_values")
        home['hems'] = redis_client.hgetall("hems_values")
        home['hems']["weekday_occ_schedule"] = [[19,8],[17,18]]
        if 'battery' in home['type']:
            home['battery'] = redis_client.hgetall("battery_values")
        if 'pv' in home['type']:
            home['pv'] = redis_client.hgetall("pv_values")
        home['wh']['draw_sizes'] = [float(i) for i in redis_client.lrange('draw_sizes', 0, -1)]
        home['hems']['weekday_occ_schedule'] = redis_client.lrange('weekday_occ_schedule', 0, -1)
        print(f"Welcome {home['name']}")

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
            if state in self.home.optimal_vals.keys():
                obs += [self.home.optimal_vals[state]]
                self.obs_dict.update({state:self.home.optimal_vals[state]})
            elif state == "leaving_horizon":
                obs += [self.home.index_8am[0] if self.home.index_8am else -1]
                self.obs_dict.update({state:self.home.index_8am[0] if self.home.index_8am else -1})
            elif state == "returning_horizon":
                obs += [self.home.index_5pm[0] if self.home.index_5pm else -1]
                self.obs_dict.update({state:self.home.index_5pm[0] if self.home.index_5pm else -1})
            elif state == "occupancy_status":
                obs += [int(self.home.occ_slice[0])]
                self.obs_dict.update({state:int(self.home.occ_slice[0])})
            elif state == "future_waterdraws":
                obs += [self.home.draw_frac.value]
                self.obs_dict.update({state:self.home.draw_frac.value})
            elif state == "oat_future":
                obs += [self.home.oat_current[-1]]
                self.obs_dict.update({state:self.home.oat_current[-1]})
            elif state == "oat_current":
                obs += [self.home.oat_current[0]]
                self.obs_dict.update({state:self.home.oat_current[-1]})
            elif state == "time_of_day":
                tod = self.home.timestep % (24 * self.home.dt)
                obs += [tod]
                self.obs_dict.update({state:tod})
            elif state == "community_demand":
                community_demand = self.home.redis_client.hget("current_values", "current_demand")
                obs += [community_demand]
                self.obs_dict.update({state:community_demand})
            elif state == "my_demand":
                obs += [self.home.stored_optimal_vals["p_grid_opt"][0]]
                self.obs_dict.update({state:self.home.stored_optimal_vals["p_grid_opt"][0]})
            else:
                print(f"MISSING {state}")

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
        kpis = {"std_demand": np.std(self.demand_profile), "max_demand": np.max(self.demand_profile)}

        with open("score.txt", 'w'):
            kpis.write(json.dumps(kpis))

        return kpis

    def step(self, action=None):
        """
        :input: action (list of floats)
        Redefines the OpenAI Gym environment step.
        :return: observation (list of floats), reward (float), is_done (bool), debug_info (set)
        """
        self.nstep += 1
        if not os.path.isdir("home_logs"):
            os.mkdir("home_logs")
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        fh.setLevel(logging.WARN)

        self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        self.redis_client = rc.connection(self.redis_url)#RedisClient()
        self.home.redis_get_initial_values()
        self.home.cast_redis_timestep()

        if self.home.timestep > 0:
            self.home.redis_get_prev_optimal_vals()

        self.home.get_initial_conditions()

        self.home.add_type_constraints()
        if action is not None:
            if "ev_charge" in self.actions:
                self.home.override_ev_charge(action[-1]) # overrides the p_ch for the electric vehicle
                action = action[:-1]
            if "wh_setpoint" in self.actions:
                self.home.override_t_wh(action[-1]) # same but for waterheater
                action = action[:-1]
            if "hvac_setpoint" in self.actions:
                self.home.override_t_in(action[-1]) # changes thermal deadband to new lower/upper bound
        
        self.home.set_type_p_grid()
        self.home.solve_mpc(debug=True)
        self.home.cleanup_and_finish()
        self.home.redis_write_optimal_vals()
        # self.home.run_home()

        self.home.log.removeHandler(fh)

        asyncio.run(self.post_status("updated"))
        asyncio.run(self.await_status("forward"))

        self.demand_profile += [self.home.stored_optimal_vals["p_grid_opt"]]

        return self.get_obs(), self.get_reward(), False, {}

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
        async_redis = aioredis.from_url(self.redis_url)
        pubsub = async_redis.pubsub()
        await pubsub.subscribe("channel:1")
        print(f"{self.home.name} {status} at t = {self.nstep}.")
        await async_redis.publish("channel:1", f"{self.home.name} {status} at t = {self.nstep}.")
        return 

if __name__=="__main__":
    import random 
    tic = datetime.now()
    my_home = PlayerHome()

    for _ in range(my_home.num_timesteps * my_home.home.dt):
        action = [random.uniform(16,22), random.uniform(0,1)]
        my_home.step(action) 

    asyncio.run(my_home.post_status("done"))
    print(my_home.score())
    toc = datetime.now()
    print(toc-tic)
