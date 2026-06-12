FROM hhcmhub/xbot2-noble-robot

ARG ROS_IP

RUN if [ -z "$ROS_IP" ]; then echo "ROS_IP is not set"; exit 1; fi

RUN git clone https://github.com/advrhumanoids/ros2_config.git && \
    cd ros2_config && \
    mkdir build && cd build && \
    cmake .. -DROS2_IP=${ROS_IP}

RUN echo "source ~/ros2_config/dds/cyclonedds/setup.bash" >> ~/.bashrc

RUN sudo apt update && \
    sudo apt install -y ros-${ROS_DISTRO}-realsense2-camera && \
    sudo rm -rf /var/lib/apt/lists/*
