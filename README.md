# dragg-comp
This is the version required for utilizing the DRAGG package (GitHub.com/apigott/dragg) as a multiplayer reinforcement learning game

The architecture allows for a player (player.py) to interface with the aggregator (rl_aggregator.py) locally or over a shared server. 

# Installation
The competition version of DRAGG can be installed via pip using `pip install dragg-comp`. 

A sample of the game can be run via `run_submission.py` which (1) starts a Redis server on the local host and then runs both `rl_aggregator.py` and `player.py`. 
