tmux kill-session -t pfvtr

tmux new-session -d -s "pfvtr" -n "pfvtr"

tmux new-window -d -n "mapping"
tmux new-window -d -n "repeating"


x=$(echo $SHELL | sed 's:.*/::')

tmux send-keys -t pfvtr:pfvtr "source ./install/local_setup.$x" Enter
tmux send-keys -t pfvtr:pfvtr "sleep 2" Enter
tmux send-keys -t pfvtr:pfvtr "ros2 launch pfvtr sim.launch.py" Enter

tmux send-keys -t pfvtr:mapping "source ./install/local_setup.$x" Enter
tmux send-keys -t pfvtr:mapping 'ros2 action send_goal /pfvtr/mapmaker pfvtr/action/MapMaker "{save_imgs_for_viz: true, map_name: '"'"'my_first_map'"'"', start: true, map_step: 1.0, source_map: '"'"''"'"', record_backward: false}"'

tmux send-keys -t pfvtr:repeating "source ./install/local_setup.$x" Enter
tmux send-keys -t pfvtr:repeating 'ros2 action send_goal /pfvtr/repeater pfvtr/action/MapRepeater "{start_pos: 0.0, end_pos: 0.0, traversals: 1, null_cmd: true, image_pub: 0, use_dist: true, map_name: '"'"''"'"'}"'
