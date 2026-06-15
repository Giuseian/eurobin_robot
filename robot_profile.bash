SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export ROS_IP=10.24.4.230
export ROS_MASTER_URI=http://10.24.4.100:11311

source $SCRIPT_DIR/robot_docker/setup.sh
