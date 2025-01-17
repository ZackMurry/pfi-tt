#!/bin/bash

source /root/.ap-set-experiment-env.sh
source /root/.bashrc


export AERPAW_REPO=${AERPAW_REPO:-/root/AERPAW-Dev}
export AERPAW_PYTHON=${AERPAW_PYTHON:-python3}
export PYTHONPATH=/usr/local/lib/python3/dist-packages/
export EXP_NUMBER=${EXP_NUMBER:-1}

if [ "$AP_EXPENV_THIS_CONTAINER_NODE_VEHICLE" == "vehicle_uav" ]; then
    export VEHICLE_TYPE=drone
elif [ "$AP_EXPENV_THIS_CONTAINER_NODE_VEHICLE" == "vehicle_ugv" ]; then
    export VEHICLE_TYPE=rover
else
    export VEHICLE_TYPE=none
fi

if [ "$AP_EXPENV_SESSION_ENV" == "Virtual" ]; then
    export LAUNCH_MODE=EMULATION
elif [ "$AP_EXPENV_SESSION_ENV" == "Testbed" ]; then
    export LAUNCH_MODE=TESTBED
else
    export LAUNCH_MODE=none
fi

export RESULTS_DIR="${RESULTS_DIR:-/root/Results}"
export TS_FORMAT="${TS_FORMAT:-'[%Y-%m-%d %H:%M:%.S]'}"
export LOG_PREFIX="$(date +%Y-%m-%d_%H_%M_%S)"

export PROFILE_DIR=$AERPAW_REPO"/AHN/E-VM/Profile_software"
#cd $PROFILE_DIR"/ProfileScripts"

cd /root

rm -f /root/screenlog.0
rm -f /root/drone.log

# screen -L -S drone -dm \
#     python3 -m aerpawlib --script dqn_drone --vehicle drone --conn :14550
screen -L -Logfile /root/drone.log -S drone -dm \
    python3 -m aerpawlib --script dqn_rover --vehicle drone --conn :14550 \
            --zmq-identifier DRONE --zmq-proxy-server 192.168.130.2 \
            --script dqn_drone
#./Radio/startRadio.sh
#./Traffic/startTraffic.sh
#./Vehicle/startVehicle.sh
