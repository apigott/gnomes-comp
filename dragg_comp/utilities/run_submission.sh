# redis-cli shutdown
# redis-server --daemonize yes
cd ..
python rl_aggregator.py &
python player.py &
# redis-cli shutdown