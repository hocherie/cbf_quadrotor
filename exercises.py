import numpy as np
import matplotlib.pyplot as plt
import ecbf_control
from ecbf_control import Robot_Sim
import warnings

warnings.filterwarnings("ignore")

def main():
    #! EXERCISE 1: FILL OUT ECBF_CONTROL.PY compute_safe_control()
    
    ### Define Robot 0
    x_init0 = np.array([5, -5, 10])
    goal_init0 =np.array([[-5], [5]])
    Robot0 = Robot_Sim(x_init0, goal_init0, robot_id=0)

    ### Define Robot 1
    x_init1 =np.array([-5, 5, 10])
    goal_init1 =np.array([[5], [-5]])
    Robot1 = Robot_Sim(x_init1, goal_init1, robot_id=1)

    #! EXERCISE 2: ADD 2 More Robots (4 in Total). Don't overlap + ADD some obstacles


    #! EXERCISE 3: CREATE DEADLOCK WITH 4 ROBOTS (No Obstacle), FIX IT


    Robots = [Robot0, Robot1] #! E2: Append with new robots

    

    a, ax1 = plt.subplots()
    
    ## Define Obstacles
    # obs1 = np.array([[2], [2]])
    # obs2 = np.array([[-2], [-2]])

    # obs = np.hstack((obs1, obs2)).T 
    obs = []   

    for tt in range(20000):

        obstacles = []
        for robot in Robots:
            obstacles.append(robot.update_obstacles(Robots, obs, noisy=False))

        u_hat_acc = []
        for robot in Robots:
            u_hat_acc.append(robot.robot_step(np.array(obstacles[robot.id]["obs"])[:, :, 0].T, np.array(obstacles[robot.id]["obs_v"])[:, :, 0].T))

        if(tt % 10 == 0):
            print(tt)
            plt.cla()
            sz = 0
            p = []
            x = 0
            y = 0
            z = 0
            for robot in Robots:
                ecbf_control.plot_step(robot.id, robot.ecbf, np.array(obstacles[robot.id]["obs"])[:, :, 0].T, u_hat_acc[robot.id], robot.state_hist, ax1)
                p.append( robot.ecbf.compute_plot_z(np.array(obstacles[robot.id]["obs"])[:, :, 0].T) )
                x = x + p[robot.id]["x"]
                y = y + p[robot.id]["y"]
                z = z + p[robot.id]["z"]
                
                sz = sz + 1
                 
            Robot0.ecbf.plot_h(x/sz, y/sz, z/sz)
            plt.pause(0.00000001)

if __name__=="__main__":
    main()
