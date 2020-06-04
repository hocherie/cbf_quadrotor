
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
    robbie = Robot(map1)

    for i in range(100):
        print("Time " + str(i))
        plt.cla()
        
        robbie.update()
        map1.visualize_map()
        robbie.visualize()
        plt.pause(0.1)
        
    print("done!!")

if __name__ == '__main__':
    main()
