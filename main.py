
from simulator import Map, LidarSimulator, Robot
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def main():
    print("start!!")

    # load map
    src_path_map = "data/two_obs.dat"
    map1 = Map(src_path_map)

    # initialize robot (initializes lidar with map) 
    rob1 = Robot(map1)
    rob2 = Robot(map1)

    rob2.state["x"][0] = 10
    rob2.state["x"][1] = 10

    for i in range(100):
        print("Time " + str(i))
        plt.cla()
        
        rob1.update()
        map1.visualize_map()
        rob1.visualize()

        rob2.update()
        map1.visualize_map()
        rob2.visualize()
        plt.pause(0.1)
        
    print("done!!")

if __name__ == '__main__':
    main()
