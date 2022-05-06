# vehicle-path-guidance

![Animation](https://github.com/leet4th/vehicle-path-guidance/blob/main/animation.gif)

<img src="https://github.com/leet4th/vehicle-path-guidance/blob/main/animation.gif" width="600" height="600" />

### Goal:
- Develp a steering solution that allows a vehicle to follow a stright line, perform a U-Turn, and then follow the line again in the opposite direction

### Assumptions:
- Kinematic Bicycle Model
  - Surface is perfectly planer/flat
  - Front and Rear wheels connected by rigid link of fixed length
  - Front wheels are steerable and act together
  - Pure rolling constraint
    - No slip or skidding
  - Non holonomic constraint
    - Move only along direction of heading, no lateral movements
- Longitudinal control
  - Perfect velocity control, only concerned with lateral control for this exercise
- Steering angle is only control input to plant
- No limitations on steering command rate of change
- No restriction how available space outside of line
- Parameters:
  - Wheel Base = 2.5 m
  - Max Steering Angle = 30 deg
  - Velocity Magnitude = 20 kph