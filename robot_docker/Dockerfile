FROM hhcmhub/xbot2-noble-robot

ARG ROS_IP

RUN if [ -z "$ROS_IP" ]; then echo "ROS_IP is not set"; exit 1; fi

RUN sudo apt update && \
    sudo apt install -y \
        git \
        cmake \
        python3-pip \
        ros-${ROS_DISTRO}-realsense2-camera && \
    sudo rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/advrhumanoids/ros2_config.git /home/user/ros2_config && \
    cd /home/user/ros2_config && \
    mkdir build && cd build && \
    cmake .. -DROS2_IP=${ROS_IP}

RUN git clone https://github.com/Giuseian/PoseEstimation.git /home/user/PoseEstimation && \
    cd /home/user/PoseEstimation && \
    git submodule update --init --recursive

RUN git clone https://github.com/Giuseian/eurobin.git /home/user/eurobin && \
    cd /home/user/eurobin && \
    git submodule update --init --recursive

RUN echo "source /home/user/ros2_config/dds/cyclonedds/setup.bash" >> /home/user/.bashrc && \
    echo "export EUROBIN_WS=/home/user/eurobin" >> /home/user/.bashrc && \
    echo "export POSE_ESTIMATION_WS=/home/user/PoseEstimation" >> /home/user/.bashrc && \
    echo "export SHARED_DATA=/home/user/shared_data" >> /home/user/.bashrc

WORKDIR /home/user