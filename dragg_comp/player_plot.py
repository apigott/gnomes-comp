# import os
# import sys
# import json
# import toml
# from datetime import datetime, timedelta
# import numpy as np
# import pandas as pd
# import itertools as it
# import random

# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
import plotly.io as pio
import plotly

# from dragg.logger import Logger
from dragg.plot import Plotter

class PlayerPlotter(Plotter):
	def __init__(self, res_file='outputs/', conf_file='outputs/all_homes-10-config.json'):
		super().__init__(res_file, conf_file)

	def plot_soc(self, name="PLAYER"):
		fig = super().plot_soc()
		pio.write_image(fig, "outputs/PLAYER-soc.png")
		return 

	def plot_community_peak(self):
		fig = super().plot_community_peak()
		pio.write_image(fig, "outputs/community-peak.png")
		return 

	def main(self):
		self.plot_soc()
		self.plot_community_peak()
