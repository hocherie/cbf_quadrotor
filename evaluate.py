"""
evaluate.py

Contains functions to evaluate safe control methods. Should contain multiple metrics.

"""
from simulator import Map, LidarSimulator, Robot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import random

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def distance_to_closest_obstacle(dense_lidar, robot):
    """Input: state, maps.
       Output: closest distance (m)
    """
    dense_lidar.update_reading((robot.x, robot.y))
    return np.min(dense_lidar.ranges)

def main():

    # Instantiate Map
    src_path_map = "data/two_obs.dat"
    map1 = Map(src_path_map)

    # Instantiate dense lidar for evaluation
    dense_lidar = LidarSimulator(map1, angles=np.arange(90)*4)

    # Instantiate Robot to be evaluated
    safe_robbie = Robot(map1, use_safe=True)
    unsafe_robbie = Robot(map1, use_safe=False)

    # Instantiate list to store closest distance over time
    safe_closest_list = []
    unsafe_closest_list = []

    for i in range(100):

        plt.cla()

        # Move robot
        safe_robbie.update()
        unsafe_robbie.update()

        # Evaluation: Get distance to closest obstacle w/ dense lidar
        safe_closest = distance_to_closest_obstacle(dense_lidar, safe_robbie)
        safe_closest_list.append(safe_closest)
    
        unsafe_closest = distance_to_closest_obstacle(dense_lidar, unsafe_robbie)
        unsafe_closest_list.append(unsafe_closest)
        # TODO: write to text file

        # # Visualize
        # map1.visualize_map()
        # safe_robbie.visualize()
        # plt.pause(0.1)


    # Visualize history
    map1.visualize_map()
    safe_robbie.visualize_robot()
    unsafe_robbie.visualize_robot()
    custom_legend = [Line2D([0], [0], color=color_cycle[0], lw=4),
                Line2D([0], [0], color=color_cycle[1], lw=4)]
    plt.legend(custom_legend, ["Safe Control", "Unsafe Control"])

    # Plot Closest Distance over Time
    plt.figure()
    plt.plot(range(len(safe_closest_list)), safe_closest_list, label="Safe")
    plt.plot(range(len(unsafe_closest_list)), unsafe_closest_list, label="Unsafe")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Distance to Closest Obstacle (m)")
    plt.show()

if __name__ == '__main__':
    main()