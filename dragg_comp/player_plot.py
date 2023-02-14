import plotly.io as pio
import plotly

from dragg.plot import Plotter

import numpy as np
import os 

class PlayerPlotter(Plotter):
	def __init__(self, res_file='outputs/', conf_file='outputs/all_homes-10-config.json'):
		super().__init__(res_file, conf_file)

	def plot_soc(self, name="PLAYER"):
		fig = super().plot_soc(name)#.show()
		pio.write_image(fig, f"outputs/{name}-soc.png")
		return 

	def plot_community_peak(self):
		fig = super().plot_community_peak()#.show()
		pio.write_image(fig, "outputs/community-peak.png")
		return 

	def check_scores(self):
		l2norm = np.linalg.norm(self.data["PLAYER"]["p_grid_opt"])
		community_load = np.sum([self.data[i]["p_grid_opt"] if i != "Summary" else np.zeros(len(self.data["PLAYER"]["p_grid_opt"])) for i in self.data.keys()], axis=0)
		i = np.argmax(community_load)
		load_at_peak = self.data["PLAYER"]["p_grid_opt"][i]
		contribution2peak = load_at_peak / community_load[i]
		print(l2norm, contribution2peak)

	def main(self):
		self.plot_soc()
		self.plot_community_peak()
		self.check_scores()
