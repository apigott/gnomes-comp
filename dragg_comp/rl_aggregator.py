import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# utilities
import os
import sys
from copy import copy, deepcopy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# redis with asynchronous utilities
import redis
import asyncio
import aioredis
import async_timeout

# pathos for parallelization
import pathos
from pathos.pools import ProcessPool

# Local
from dragg.mpc_calc import MPCCalc, manage_home
from dragg.aggregator import Aggregator
# from dragg.redis_client import RedisClient
import dragg.redis_client as rc
from dragg.logger import Logger

REDIS_URL = "redis://localhost"

class RLAggregator(Aggregator):
    def __init__(self, start=None, end=None, redis_url=REDIS_URL):
        self.redis_url = redis_url
        super().__init__(start, end, redis_url)
        self.overwrite_output = True
        self.mpc_players = [] # RLAggregator distinguishes between comp controlled (mpc_players) and human players
        self.case = 'gnomes'

    def get_home_names(self):
        all_home_names = super().get_home_names()
        all_home_names[0] = "PLAYER"
        return all_home_names

    def get_hems_params(self, override_horizon=False):
        """
        Returns a set of parameters for the home energy management system.
        """
        responsive_hems = super().get_hems_params()
        if override_horizon:
            responsive_hems["horizon"] = 1
        return responsive_hems

    def get_homes(self):
        """
        Creates randomized home parameters for R1C1 thermal models.
        :return: None
        """
        homes_file = os.path.join(self.outputs_dir, f"all_homes-{self.config['community']['total_number_homes']}-config.json")
        if not self.config['community']['overwrite_existing'] and os.path.isfile(homes_file):
            with open(homes_file) as f:
                self.all_homes = json.load(f)
        else:
            self.create_homes()
            self.all_homes_copy = copy(self.all_homes)
        self._check_home_configs()
        self.write_home_configs()
        return

    def post_next_home(self, initialize_mpc=False):
        """
        Posts parameters for an MPC (linearized R1C1 home) to the Redis server.
        :return: None
        """
        if not initialize_mpc:
            if len(self.all_homes_copy) > 0:
                next_home = self.all_homes_copy.pop()
            else:
                self.log.logger.warn("You have initialized more players than are set in the community")

            self.redis_client.hset("simulation", "nsteps", self.num_timesteps)
            for k, v in next_home.items():
                if not k in ["wh","hvac","battery","pv","hems"]:
                    self.redis_client.hset("home_values", k, v)
                else:
                    for k2, v2 in v.items():
                        if not k2 in ["draw_sizes", "weekday_occ_schedule"]:
                            self.redis_client.hset(f"{k}_values", k2, v2)
                        else:
                            self.redis_client.delete(k2)
                            self.redis_client.rpush(k2, *v2)

        else:
            for next_home in self.all_homes_copy:
                self.mpc_players += [MPCCalc(next_home)]
                self.log.logger.info(f"Aggregator initialized MPC player {self.mpc_players[-1].name}")

        return 

    async def post_status(self, status,  channel: aioredis.client.PubSub, redis_client):
        """
        :input: Status (string)
        Publishes a status (typically "is done" to alert the aggregator)
        :return: None
        """
        # redis = aioredis.from_url(self.redis_url)
        # pubsub = redis.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")
        await redis.publish("channel:1", status)
        return 

    async def await_player(self, channel: aioredis.client.PubSub, redis_client):
        i = 0
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await channel.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        if "initialized player" in message["data"].decode():
                            self.log.logger.debug("Initialized a new player")
                            i += 1 
                            if i < self.config['community']['n_players']:
                                self.post_next_home()
                                i += 1
                                
                            elif i == self.config['community']['n_players']: # now we know that the whole community has stepped
                                i = 0
                                self.post_next_home(initialize_mpc=True)
                                await redis_client.publish("channel:1", "all ready")
                                break
                    await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                pass
        return 

    async def reader(self, channel: aioredis.client.PubSub, redis_client):
        """
        Opens an asynchronous subscription to the specified PubSub channel on Redis. Awaits player 
        controlled homes to announce that they've finished one timestep (or completed their initialization)
        and made a control decision and implement it, then implements all computer controlled actions
        simaultaneously.
        :return None:
        """

        i = 1
        self.next_ts = 1
        while True:
            try:
                async with async_timeout.timeout(3):
                    message = await channel.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        if str(self.next_ts) in message["data"].decode():
                            if "updated" in message["data"].decode():
                                self.log.logger.info(f"Player {i}/{self.config['community']['n_players']} updated demand at t={self.next_ts}.")
                                if i == self.config['community']['n_players']: # now we know that the whole community has stepped
                                    self.redis_set_current_values()
                                    self.run_iteration()
                                    await redis_client.publish("channel:1", "timestep can be moved forward")
                                    self.collect_data()
                                    i = 0
                                    self.next_ts += 1
                                i += 1

                        elif "done" in message["data"].decode():
                            self.write_outputs()
                            return # fix this break

                        elif "reset" in message["data"].decode():
                            self.timestep = 0
                            self.next_ts = 1

                    await asyncio.sleep(0.1)
            
            except asyncio.TimeoutError:
                self.log.logger.info("TIMEOUT (No update from MPC players yet.)")
                pass

        return

    async def open_server(self):
        """
        Runs simulation(s) specified in the config file with all combinations of
        parameters specified in the config file.
        :return: None
        """
        self.log.logger.info("Made it to Aggregator Run")
        self.checkpoint_interval = 500 # default to checkpoints every 1000 timesteps
        if self.config['simulation']['checkpoint_interval'] == 'hourly':
            self.checkpoint_interval = self.dt
        elif self.config['simulation']['checkpoint_interval'] == 'daily':
            self.checkpoint_interval = self.dt * 24
        elif self.config['simulation']['checkpoint_interval'] == 'weekly':
            self.checkpoint_interval = self.dt * 24 * 7

        # set the version of the run for storing output files
        self.version = self.config['simulation']['named_version']
        self.set_run_dir()
        self.case = "baseline" # no aggregator level control
        
        # flush the database that we read/write to
        self.flush_redis()
        self.get_homes()
        self.post_next_home()
        self.reset_collected_data()

        # everything is ready to begin the simulation (let players know)
        self.log.logger.info("Starting aioredis listener...")
        redis = aioredis.from_url(self.redis_url)
        pubsub = redis.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")
        await redis.publish("channel:1", "ready")

        # initialize all players with their params (write to redis, players read)
        await asyncio.create_task(self.await_player(pubsub, redis))

        # creates the main loop for updating at each and every timestep
        await asyncio.create_task(self.reader(pubsub, redis))

if __name__=="__main__":
    a = RLAggregator(sys.argv[1], sys.argv[2])
    asyncio.run(a.open_server())
