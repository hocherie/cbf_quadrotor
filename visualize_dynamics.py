from sim_utils import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def visualize_quad_quadhist(ax, quad_hist, t):
    """Works with QuadHist class."""
    visualize_quad(ax, quad_hist.hist_x[:t], quad_hist.hist_y[:t],
                   quad_hist.hist_z[:t], quad_hist.hist_pos[t], quad_hist.hist_theta[t])


def visualize_error_quadhist(ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error, quad_hist,t, dt):
    """Works with QuadHist class."""
    visualize_error(ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error,
                    quad_hist.hist_pos[:t+1], quad_hist.hist_xdot[:t+1], quad_hist.hist_theta[:t+1], quad_hist.hist_des_theta[:t+1], quad_hist.hist_thetadot[:t+1], dt, quad_hist.hist_des_xdot[:t+1], quad_hist.hist_des_x[:t+1],
                    quad_hist.hist_xdotdot[:t+1])

def animate_quad(ax, hist_x, hist_y, hist_z, cur_state, cur_theta):
    """Plot quadrotor 3D position and history"""
    x = cur_state
    theta = np.radians(cur_theta)
    R = get_rot_matrix(theta)
    plot_L = 1
    quad_ends_body = np.array(
        [[-plot_L, 0, 0], [plot_L, 0, 0], [0, -plot_L, 0], [0, plot_L, 0], [0, 0, 0], [0, 0, 0]]).T
    quad_ends_world = np.dot(R, quad_ends_body) + np.matlib.repmat(x, 6, 1).T
    # Plot Rods
    ax.plot3D(quad_ends_world[0, 0:2],
              quad_ends_world[1, 0:2], quad_ends_world[2, 0:2], 'r')
    ax.plot3D(quad_ends_world[0, 2:4],
              quad_ends_world[1, 2:4], quad_ends_world[2, 2:4], 'b')
    # Plot drone center
    ax.scatter3D(x[0], x[1], x[2], edgecolor="r", facecolor="r")

    # Plot history
    ax.scatter3D(hist_x, hist_y, hist_z, edgecolor="b",
                 facecolor="b", alpha=0.1)
    # ax_th_error.
    ax.set_xlim(x[0]-3, x[0]+3)
    ax.set_ylim(x[1]-3, x[1]+3)
    ax.set_zlim(x[2]-5, x[2]+5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.pause(0.1)

def visualize_quad(ax, hist_x, hist_y, hist_z, cur_state, cur_theta):
    """Plot quadrotor 3D position and history"""
    x = cur_state
    theta = np.radians(cur_theta) 
    R = get_rot_matrix(theta)
    plot_L = 1
    quad_ends_body = np.array(
        [[-plot_L, 0, 0], [plot_L, 0, 0], [0, -plot_L, 0], [0, plot_L, 0], [0, 0, 0], [0, 0, 0]]).T
    quad_ends_world = np.dot(R, quad_ends_body) + np.matlib.repmat(x, 6, 1).T
    # Plot Rods
    ax.plot3D(quad_ends_world[0, [1,5]],
              quad_ends_world[1, [1,5]], quad_ends_world[2, [1,5]], 'r') # body x front
    ax.plot3D(quad_ends_world[0, [0,5]],
              quad_ends_world[1, [0,5]], quad_ends_world[2, [0,5]], 'k') # body x back
    # ax.plot3D(quad_ends_world[0, 0:2],
    #           quad_ends_world[1, 0:2], quad_ends_world[2, 0:2], 'r') # body x
    ax.plot3D(quad_ends_world[0, 2:4],
              quad_ends_world[1, 2:4], quad_ends_world[2, 2:4], 'b') # body y
    # Plot drone center
    ax.scatter3D(x[0], x[1], x[2], edgecolor="r", facecolor="r")

    # Plot history
    ax.scatter3D(hist_x, hist_y, hist_z, edgecolor="b",
                 facecolor="b", alpha=0.1)
    # ax_th_error.
    ax.set_xlim(x[0]-3, x[0]+3)
    ax.set_ylim(x[1]-3, x[1]+3)
    ax.set_zlim(x[2]-5, x[2]+5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.pause(0.1)


def visualize_error(ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error, hist_pos, hist_xdot, hist_theta, hist_des_theta, hist_thetadot, dt, hist_des_xdot, hist_des_x, hist_xdotdot):
    # pass
    # ax.plot([0,1], [1,10],'b')

    # Position Error
    ax_x_error.plot(np.array(range(len(hist_theta))) *
                    dt, np.array(hist_pos)[:, 0], 'k')
    ax_x_error.plot(np.array(range(len(hist_theta))) *
                    dt, np.array(hist_pos)[:, 1], 'b')
    ax_x_error.plot(np.array(range(len(hist_theta))) *
                    dt, np.array(hist_pos)[:, 2], 'r')
    # Desired Pos
    ax_x_error.plot(np.array(range(len(hist_theta))) *
                    dt, np.array(hist_des_x)[:, 0], 'k--')
    ax_x_error.plot(np.array(range(len(hist_theta))) *
                    dt, np.array(hist_des_x)[:, 1], 'b--')
    ax_x_error.plot(np.array(range(len(hist_theta))) *
                    dt, np.array(hist_des_x)[:, 2], 'r--')
    ax_x_error.set_title("Position (world)")
    ax_x_error.legend(["x", "y", "z"])

    # TODO: make into funciton for each plot
    # Velocity Error
    ax_xd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_xdot)[:, 0], 'k')
    ax_xd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_xdot)[:, 1], 'b')
    ax_xd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_xdot)[:, 2], 'r')
    # Desired Velocity
    ax_xd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_des_xdot)[:, 0], 'k--')
    ax_xd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_des_xdot)[:, 1], 'b--')
    ax_xd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_des_xdot)[:, 2], 'r--')
    ax_xd_error.legend(["x", "y", "z"])
    ax_xd_error.set_title("Velocity (world)")

    # Angle Error
    ax_th_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_theta)[:, 0], 'k')
    ax_th_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_theta)[:, 1], 'b')
    ax_th_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_theta)[:, 2], 'r')
    # Desired angle
    ax_th_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_des_theta)[:, 0], 'k--')
    ax_th_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_des_theta)[:, 1], 'b--')
    ax_th_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_des_theta)[:, 2], 'r--')

    ax_th_error.legend(["Roll", "Pitch", "Yaw"])
    ax_th_error.set_ylim(-40, 40)
    ax_th_error.set_title("Angle")

    # Angle Rate
    ax_thr_error.plot(np.array(range(len(hist_theta))) *
                      dt, np.array(hist_thetadot)[:, 0], 'k')
    ax_thr_error.plot(np.array(range(len(hist_theta))) *
                      dt, np.array(hist_thetadot)[:, 1], 'b')
    ax_thr_error.plot(np.array(range(len(hist_theta))) *
                      dt, np.array(hist_thetadot)[:, 2], 'r')
    # ax.plot(range(len(hist_theta)), np.array(des_theta)[:, 0])
    ax_thr_error.legend(["Roll Rate", "Pitch Rate", "Yaw Rate"])
    ax_thr_error.set_ylim(-100, 100)
    
    ax_thr_error.set_title("Angular Rate")


    # Acceleration
    ax_xdd_error.plot(np.array(range(len(hist_theta))) *
                     dt, np.array(hist_xdotdot)[:, 0], 'k')
    ax_xdd_error.plot(np.array(range(len(hist_theta))) *
                      dt, np.array(hist_xdotdot)[:, 1], 'b')
    ax_xdd_error.plot(np.array(range(len(hist_theta))) *
                      dt, np.array(hist_xdotdot)[:, 2], 'r')
    ax_xdd_error.legend(["x", "y", "z"])
    ax_xdd_error.set_title("Acc. (world)")
    
    plt.pause(0.1)
