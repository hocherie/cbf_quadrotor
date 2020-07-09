import numpy as np
import matplotlib.pyplot as plt
import ecbf_control
from ecbf_control import Robot_Sim

def main():
    
    ### Robot 1
    x_init0 = np.array([3, -5, 10])
    goal_init0 =np.array([[-6], [4]])
    Robot0 = Robot_Sim(x_init0, goal_init0, robot_id=0)

    ### Robot 2

    x_init1 =np.array([-5, 3, 10])
    goal_init1 =np.array([[4], [-6]])
    Robot1 = Robot_Sim(x_init1, goal_init1, robot_id=1)


    ### Robot 3

    x_init2 =np.array([3, 5, 10])
    goal_init2 =np.array([[-3], [-7]])
    Robot2 = Robot_Sim(x_init2, goal_init2, robot_id=2)

    ### Robot 4

    x_init3 =np.array([-5, -3, 10])
    goal_init3 =np.array([[7], [5]])
    Robot3 = Robot_Sim(x_init3, goal_init3, robot_id=3)

    # ### Robot 5

    # x_init5 =np.array([5, 0, 10])
    # goal_init5 =np.array([[-6], [0]])
    # Robot5 = Robot_Sim(x_init5, goal_init5, 4)
    
    Robots = [Robot0, Robot0, Robot3, Robot4]

    plt.plot([2, 2, 3])

    a, ax1 = plt.subplots()
    
    ## Obstacles
    # const_obs = np.array([[2], [2]])
    # const_obs2 = np.array([[-2], [-2]])

    # obs = np.hstack((const_obs2, const_obs)).T    
    obs = []

    for tt in range(20000):

        obstacles = []
        for robot in Robots:
            obstacles.append(robot.update_obstacles(Robots, obs))

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
                # start_time = time.time()
                ecbf_control.plot_step(robot.id, robot.ecbf, np.array(obstacles[robot.id]["obs"])[:, :, 0].T, u_hat_acc[robot.id], robot.state_hist, ax1)
                # proc2_time = time.time()
                # print("Time Elapsed (plot_step)", proc2_time - start_time)
                
                
                p.append( robot.ecbf.compute_plot_z(np.array(obstacles[robot.id]["obs"])[:, :, 0].T) )
                # proc3_time = time.time()
                # print("Time Elapsed (compute_plot_z)", proc3_time - proc2_time)
                x = x + p[robot.id]["x"]
                y = y + p[robot.id]["y"]
                z = z + p[robot.id]["z"]
                
                sz = sz + 1
            # start_time = time.time()
            Robot2.ecbf.plot_h(x/sz, y/sz, z/sz)
            # proc2_time = time.time()
            # print("Time Elapsed (plot_H)", proc2_time - start_time)
            plt.pause(0.00000001)

if __name__=="__main__":
    main()
