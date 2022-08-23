# dragg-comp
This is the version required for utilizing the [DRAGG package](https://github.com/apigott/dragg) as a multiplayer reinforcement learning game.

The architecture allows for a player (player.py) to interface with the aggregator (rl_aggregator.py) locally or over a shared server. The user may adapt any or all of the three home energy appliances (HVAC, WH, EV) to reduce their peak electric consumption and assist the grid with grid-level goals. 

# Installation
The competition version of DRAGG can be installed via pip using `pip install dragg-comp`. 

A sample of the game can be run via `run_submission.py` which (1) starts a Redis server on the local host and (2) runs both `rl_aggregator.py` and `player.py`. The resulting output file can be plotted with the DRAGG Reformatter.

# Important note for GNOMES4Homes players:
You do *not* need to modify any of the files in this folder/package and doing so may disqualify your agent from participating in the GNOMES competition. Please refer to the submission template located in the [official competition repository](https://github.com/CUgriffinlab/dragg-comp-submission/tree/sandbox).
