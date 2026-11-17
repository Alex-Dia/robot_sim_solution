# Autonomous Vision-Guided Robot | Webots e-puck

**3006ICT — Robotics and Computer Vision | Griffith University**

An autonomous mobile-robot project combining **A* path planning, closed-loop navigation, obstacle avoidance, and camera-based object recognition** in the Webots R2025a simulator. An e-puck robot searches eight observation stations for a mission-specific target, distinguishes it from visual distractors, and drives to the designated observation position without a predetermined target-to-station mapping.

> **Repository note:** This README documents the final three-module implementation described in the team's completed project report. The starter ZIP provided for this README contains only the original `group_project_controller.py` with unfinished `TODO` sections; replace it with the final controller and add the completed `navigation.py` and `vision.py` before publishing this README as documentation of runnable source code.

## Project overview

The simulated environment contains a 4 m × 4 m arena, eight possible observation stations (S1–S8), five navigation barriers, and distractor images. The e-puck must read the requested target from `config/assessment_mission.json`, independently determine which station to visit, navigate safely, inspect the station using its RGB camera, and continue searching until the target is confirmed. After confirmation, the robot moves to the official observation position and stops within the required 0.20 m tolerance, with a mission deadline of 240 seconds.

The system uses Webots-provided GPS and InertialUnit pose readings for localisation and raw camera images for visual detection. It does **not** rely on a fixed target-to-station lookup or Webots Camera Recognition to identify objects.

## Core architecture

```text
assessment_mission.json + project_config.json + occupancy_grid.npy
                          |
                          v
              group_project_controller.py
              Mission FSM and integration
                          |
             +------------+------------+
             |                         |
             v                         v
       navigation.py                vision.py
   A* + waypoint control       ORB + RANSAC matcher
   clearance and recovery      multi-frame confirmation
             |                         ^
             v                         |
     motors / GPS / IMU       RGB camera inspection
     proximity sensors
```

### 1. `group_project_controller.py` — mission orchestration

The **main controller** interfaces with Webots and coordinates the end-to-end mission through a finite-state machine (FSM). It initialises the e-puck's wheel motors, camera, eight proximity sensors (`ps0`–`ps7`), GPS, and InertialUnit; reads the target and station configuration; and coordinates navigation and visual inspection.

Its principal responsibilities are:

- **Mission initialisation:** Load the requested target, map and station coordinates without changing the source code for each mission.
- **Station selection:** Evaluate reachable, unvisited stations and select the next one using the A* travel cost rather than fixed ordering or straight-line distance.
- **Inspection-pose selection:** Plan a collision-free camera position that provides a sufficiently complete view of the station poster.
- **State transitions:** Coordinate route planning, waypoint travel, turning/alignment, visual inspection, unsuccessful-search continuation, recovery and final approach.
- **Target confirmation:** Use repeated camera observations to reduce decisions based on a single ambiguous frame.
- **Mission completion:** After visual confirmation, route to the station's official observation point and stop within the 0.20 m positional tolerance.

### 2. `navigation.py` — path planning and motion control

The **navigation module** turns an intended station or inspection position into safe wheel-motion commands.

- **Global path planning:** Four-connected A* search over a **40 × 40 occupancy grid** at **0.10 m per cell**, using a Manhattan-distance heuristic.
- **Obstacle handling:** Reject occupied cells and apply additional costs to traversable cells close to obstacles to favour safer routes when possible.
- **Route execution:** Convert the selected grid path into world-coordinate waypoints, using GPS position and IMU yaw to calculate heading and position errors.
- **Closed-loop steering:** Apply proportional heading correction, turn-in-place behaviour and speed reduction near tighter turns or the destination.
- **Reactive safety:** Monitor the e-puck's proximity sensors; stop, turn or replan if the robot encounters an unsafe or blocked path.

This combines global map-based navigation with local sensor feedback: A* provides the route, while the waypoint follower and proximity responses manage actual movement.

### 3. `vision.py` — visual target identification

The **vision module** determines whether a camera view contains the target named in the mission file, rather than inferring its identity from the station location.

The final documented implementation uses target-specific **ORB feature matching** with geometric verification:

1. Prepare the target reference image and extract ORB keypoints/descriptors.
2. Capture the e-puck camera image and extract candidate visual features.
3. Match descriptors using brute-force Hamming-distance KNN matching (`k=2`).
4. Filter ambiguous correspondences with Lowe's ratio test (`0.75`).
5. Estimate a homography using RANSAC (`3 px` reprojection threshold) and count consistent inliers.
6. Accept a positive frame when the match evidence meets the configured minimums; confirm a target across multiple frames instead of trusting one observation.

**Documented matching parameters**

| Parameter | Value |
|---|---:|
| Camera frame size | 160 × 120 pixels |
| Reference preprocessing | 64 × 64 grayscale |
| ORB features | 1,000 |
| ORB scale factor / levels | 1.15 / 8 |
| Lowe ratio threshold | 0.75 |
| RANSAC threshold | 3 px |
| Minimum good matches | 18 |
| Minimum RANSAC inliers | 9 |
| Minimum inlier ratio | 0.40 |
| Temporal confirmation | 3 positive frames out of 5 |

The robot first aligns at a **map-valid inspection pose** with a view of the whole poster. This is important because earlier tests showed that partial or clipped poster images could cause false negatives. A successful inspection triggers a separate path to the official stopping point. These vision settings reflect the final documented design and should be checked against the precise files committed to this repository.

## Mission workflow

```text
Start simulation
      |
Read target, map and station configuration
      |
Estimate pose (GPS + IMU)
      |
Choose lowest-cost reachable unvisited station
      |
Plan A* route to a valid inspection pose
      |
Follow waypoints + proximity safety/recovery
      |
Align camera toward station poster
      |
Capture RGB frames -> ORB/RANSAC -> multi-frame voting
      |
      +-- Target not confirmed --> Mark station visited; search again
      |
      +-- Target confirmed --> Plan route to official observation point
                                   |
                            Approach and stop
```

## Project structure

```text
3006ICT_group_project_webots/
├── config/
│   ├── assessment_mission.json         # Selected target for a run
│   └── project_config.json            # Arena, starts, stations, robot metadata
├── controllers/
│   └── group_project_controller/
│       ├── group_project_controller.py # Mission FSM and Webots integration
│       ├── navigation.py               # A* planner and waypoint follower [final code required]
│       ├── vision.py                   # ORB/RANSAC recognition [final code required]
│       └── project_utils.py            # Coordinate conversion/config helpers
├── maps/
│   ├── occupancy_grid.npy
│   └── occupancy_grid.csv
├── targets/
│   └── target_reference.png
├── textures/                       # Target and distractor image assets
├── worlds/
│   ├── training_start_A.wbt
│   ├── training_start_B.wbt
│   └── training_start_C.wbt
└── protos/
```

## Setup and run

**Requirements:** Webots R2025a, Python 3 with `numpy` and `opencv-python`, and the project assets and folder structure.

1. Install Webots R2025a and configure its Python controller interpreter with NumPy and OpenCV available. The Python `controller` module is supplied by Webots when a Python controller runs inside the simulator.
2. Place the final `group_project_controller.py`, `navigation.py` and `vision.py` together under `controllers/group_project_controller/`.
3. Open one of `worlds/training_start_A.wbt`, `training_start_B.wbt` or `training_start_C.wbt` in Webots.
4. Edit `config/assessment_mission.json` to choose the mission target, for example:

   ```json
   {"target": "camera"}
   ```

5. Run or reset the simulation. The controller should begin searching autonomously, navigate between candidate stations, inspect camera views and stop at the confirmed target's designated observation point.

Supported target labels in the provided configuration are `soda_can`, `coffee_mug`, `backpack`, `fire_extinguisher`, `camera`, `running_shoe`, `headphones`, and `wall_clock`.

## Evaluation and outcomes

The team's final project report records successful runs across **three starting configurations and eight target objects**, with the **slowest recorded completion time of 235 seconds** against the 240-second mission limit. These figures are *reported team test outcomes*, not tests reproduced from the starter archive accompanying this README.

The most consequential integration issue was the relationship between navigation and vision: reaching a nominal station pose did not guarantee a usable image. Moving recognition to a camera-aware inspection pose, and navigating to the official endpoint only after confirmation, addressed this failure mode.

## Engineering trade-offs and limitations

- **Known-map planning:** Efficient for the provided static arena, but not a general solution for unknown or dynamically changing environments.
- **GPS/IMU localisation:** Suitable in the simulator; a physical deployment would need an appropriate localisation approach.
- **Feature-based recognition:** ORB/RANSAC is interpretable and relatively lightweight, but performance depends on resolution, texture, viewpoint and lighting.
- **Camera geometry:** Full-poster visibility matters; an incorrectly chosen inspection pose can make a capable detector appear unreliable.
- **Deadline pressure:** Cost-based station ordering improves search efficiency but does not guarantee optimal completion time for every possible target placement.

## Technologies

**Python · Webots R2025a · e-puck · OpenCV · NumPy · A* · Occupancy Grids · GPS/IMU · Proportional Control · ORB · RANSAC · Finite-State Machines**
