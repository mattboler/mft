# mft
A minimal feature tracker for visual or visual-inertial systems.

The ROS interface subscribes to a ```sensor_msgs::Image``` and publishes 
a ```sensor_msgs::PointCloud``` where the channels are as follows:

1. Feature ID
2. Feature age
3. Feature u coordinate
4. Feature v coordinate

## Dependencies

* OpenCV (Written and tested with 4.2.0)
* ROS (Written and tested with ROS Noetic on Ubuntu 20.04)

## Installation

```>> git clone https://github.com/mattboler/mft```

```>> catkin build```

## Usage

```roslaunch mft <your launch file>```

## Citations

Heavy inspiration drawn from:

* [uzh-rpg/vilib](https://github.com/uzh-rpg/vilib)

@inproceedings{Nagy2020,
  author = {Nagy, Balazs and Foehn, Philipp and Scaramuzza, Davide},
  title = {{Faster than FAST}: {GPU}-Accelerated Frontend for High-Speed {VIO}},
  booktitle = {IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS)},
  year = {2020}
}

* [daniilidis-group/msckf_mono](https://github.com/daniilidis-group/msckf_mono)

* [HKUST-Aerial-Robotics/vins_mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)

@article{qin2017vins,
  title={VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator},
  author={Qin, Tong and Li, Peiliang and Shen, Shaojie},
  journal={IEEE Transactions on Robotics}, 
  year={2018},
  volume={34}, 
  number={4}, 
  pages={1004-1020}
}