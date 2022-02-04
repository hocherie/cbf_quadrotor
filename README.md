# cbf_quadrotor
<img src="docs/ecbf_single_obs.gif" width="400">

Control Barrier Functions (CBFs) for Quadrotors. Based on [hocherie/2d_grid_playground](https://github.com/hocherie/2d_grid_playground) for dynamics simulator, nominal controllers, range sensing.

Accompanying Lecture for [Air Lab Summer School 2020](https://theairlab.org/summer2020/): [PDF Slides](docs/ensuring-safety-pdf.pdf)
<!-- Includes following files: -->



<!-- * **`ecbf_control.py`**: Contains `ECBF_CONTROL()` class for Exponential Control Barrier Function (ECBF). 

* **`run_one_robot_obs.py`**: Example - Single quadrotor avoiding obstacle at center using ECBFs.
* **`run_two_robots.py`**: Example - Two quadrotor avoiding each other using ECBFs.

From [hocherie/2d_grid_playground](https://github.com/hocherie/2d_grid_playground):
* **`main.py`**: Simple example on using `simulator.py`, `controller.py` and `dynamics.py`. Quadrotor maneuvering in 2D grid with 2nd order dynamics executing naive safe control.

* **`dynamics.py`**: Contains QuadDynamics class which gives a simple 3d quadrotor dynamics given 2nd order equations of motion. Use by instantiating class `dyn=QuadDynamics()` and calling `self.step_dynamics(state, u)` to update quadrotor state. Based on http://andrew.gibiansky.com/downloads/pdf/Quadcopter%20Dynamics,%20Simulation,%20and%20Control.pdf

* **`controller.py`**: Controller-related functions. Uses cascaded PID controllers. (ex. Position, Velocity, Attitude Controller). Mainly use by calling `go_to_position(...)`, `go_to_acceleration(...)`.

* **`sim_utils.py`**: Common utility functions for simulator and dynamics. Ex. `get_rot_matrix(angles)`

* **`visualize_dynamics.py`**: Contains graphing-related functions for dynamics.py. Mainly use for tuning PID controllers.

* **`simulator.py`**: (Unused by CBF files) Creates 2D grid simulator and enables basic range sening. Contains Map class (create from txt file), Robot class (stores current and paast state, also instantiates QuadDynamics object). 

* **`evaluate.py`** : (Unused by CBF files) Contains functions to evaluate safe control methods.  -->

## Getting Started 

### Installation
Please install pyenv, if not already. [Instructions.](https://realpython.com/intro-to-pyenv/) Specifically, follow (1) build dependecies and (2) Using the pyenv-installer.


```
git clone https://github.com/hocherie/cbf_quadrotor.git     # Clone Repo
cd cbf_quadrotor                    # Navigate to folder
$ pyenv install -v 3.7.2            # Install Python
$ pyenv virtualenv 3.7.2 safety-cbf # Make Virtual Environment
$ pyenv local safety-cbf            # Activate virtual env
$ pip install -r requirements.txt   # Install dependencies
```

<!-- ### Play with Control Barrier Function Safe Control (1 Robot, 1 Obstacle)
`$ python run_one_robot_obs.py`

<img src="docs/ecbf_single_obs.gif" width="400">

Robot uses Exponential Control Barrier Functions to stay safe with respect to single obstacle at center. Minimum interventional safe control (uses control with least difference to nominal while staying safe). Uses dynamics and controllers to move.
Originally given straight input to goal (red star).


### Play with Quadrotor Dynamics
`$ python dynamics.py`

<img src="docs/3d_quad_sim.gif" width="400">

Robot moves to desired position. (set in `main()`)
Uses dynamics from second order equations of motion (acceleration, torque) from `dynamics.py`, and cascaded PID controllers for position, velocity, and dynamics inversion (check?) to compute final motor input from `controller.py`.

Code first generates trajectory then visualizes.

### Play with 2D Grid Simulator
`$ python main.py`

<img src="docs/2d_grid.gif" width="400">
Robot executes naive safe control to stay safe in 2D obstacle environment given range measurements. Calculates opposing vector to closest obstacle to repulse away. Uses dynamics and controllers to move.
Originally given straight input. -->


<!-- # TODO
- [x] add ecbf to README
- [ ] update conda yml file
- [x] rename conda environment
- [x] Start with GIF
- [x] Update README.md with new screen capture of single-robot ECBF
- [x] Update installation instructions
- [x] Split CBF functions from run file
- [x] Split single robot File and multi-robot file
- [ ] Comment all functions
- [ ] Add multi-robot ECBF (MM)
- [ ] Replace first GIF with cool multi-robot (MM)
- [ ] Links to reference papers  -->

# Accompanying Paper
"Provably Safe" in the Wild: Testing Control Barrier Functions on a Vision-Based Quadrotor in an Outdoor Environment. Presented in 2021 RSS Workshop in Robust Autonomy. [[PDF]](https://openreview.net/pdf?id=CrBJIgBr2BK)

```
@inproceedings{hoshih2020provablyinwild,
  title = {"Provably Safe" in the Wild: Testing Control Barrier Functions on a Vision-Based Quadrotor in an Outdoor Environment},
  author = {Ho, Cherie* and Shih, Katherine* and Grover, Jaskaran and Liu, Changliu and Scherer, Sebastian},
  booktitle = {RSS 2020 Workshop in Robust Autonomy},
  year = {2020},
  url = {https://openreview.net/pdf?id=CrBJIgBr2BK}
}

```

# Author
Cherie Ho (cherieh@cs.cmu.edu)

Mohammadreza Mousaei (mmousaei@andrew.cmu.edu)

Kate Shih (kshih@andrew.cmu.edu)
