# lane_detection

## Description:
Driving lane detection algorithm  (Development only). The main purpose of this package is to perform lane detection using a polynomial fitting model and various image processing techniques.

**Note:** that this algorithm is NOT completed in C++ but is completed in the `/dev` folder in python code. Progress has been made in C++ up to performing an IPM on a binary image (so maybe about ~50% of the entire process).

## Requirements:

- Ubuntu Focal Fossa
- ROS2 Foxy Fitzroy
- C++17 or higher

## To Use:

**_ROS2 package_**

To use the (currently uncompleted) ROS2 package, follow the following steps (also included in the `README.md`)

***Before Use:***

- **Make sure ALL PATHS ARE SET CORRECTLY in the launch and config files before use!**
- **These steps assume you have already created a workspace folder and a `/src` directory within it!**

***Steps:***

1.  Navigate into the `/src` directory of your workspace and clone the repo using `git clone`
2.  Navigate back into the workspace directory and source `$ source /opt/ros/foxy/setup.bash`
3.  Build package `$ colcon build` or `$ colcon build --packages-select <package_name>`
4.  Open a new terminal and source it `$ . install/setup.bash`
5.  Run launch file `$ ros2 launch <package_name> <launch_file_name>` in this case it is `$ ros2 launch lane_detection_package sliding_box_ld_launch.py`

**_Python Development_**

Under the `/dev` folder are two python scripts (one for image data, and one for video data - both using the same functions) that has the completed algorithm. To change image/video paths, locate the input paths set in the run() function near the bottom of the script