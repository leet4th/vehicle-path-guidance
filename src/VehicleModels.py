
import numpy as np

class RearWheelBicycleModel:

    def __init__(self, initial_state=None, params=None):

        if initial_state is None:
            initial_state = np.array([
                3.0,     # x
                1.0,     # y
                np.pi/2, # psi
                0.0,     # psiDot
            ],dtype=float)

        if params is None:
            params = {}
            params['dt'] = 0.1
            params['L'] = 2.5
            params['max_steer_angle_deg'] = 15.0

        self.states = np.asarray(initial_state)
        self.params = params

        self.calc_dcm()
        self.calc_wheel_points()

    def calc_dcm(self):
        heading = self.states[2]
        ch = np.cos(heading)
        sh = np.sin(heading)

        self.C_toBfromL = np.array([
            [ ch, sh],
            [-sh, ch],
        ],dtype=float)

        self.C_toLfromB = self.C_toBfromL.T

    def calc_wheel_points(self):

        x = self.states[0]
        y = self.states[1]
        rear = np.array([x,y],dtype=float)

        # L vector in body frame
        Lvec_B = np.array([self.params['L'], 0])

        # Front wheel point
        front = rear + np.dot(self.C_toLfromB, Lvec_B)

        # Front wheel
        self.rear = rear
        self.front = front

    def update(self, steer_angle, vMag):

        # Unpack params
        dt = self.params['dt']
        L = self.params['L']
        max_steer_angle = self.params['max_steer_angle_deg']*np.pi/180

        # Unpack states
        x = self.states[0]
        y = self.states[1]
        psi = self.states[2]
        psiDot = self.states[3]

        # Limit steering angle
        steer_angle = np.clip(steer_angle, -max_steer_angle, max_steer_angle)

        # Precompute sin,cos
        sPsi = np.sin(psi)
        cPsi = np.cos(psi)

        # Kinematic Equations
        vx = vMag * cPsi - L * psiDot * sPsi/2
        vy = vMag * sPsi + L * psiDot * cPsi/2

        # Update DCMs
        self.calc_dcm()

        # Update wheel points (rear, front)
        self.calc_wheel_points()

        # Euler integration to next time step
        x_new = x + vx * dt
        y_new = y + vy * dt
        psi_new = np.mod(psi + psiDot *dt, 2*np.pi)
        psiDot_new = steer_angle * vMag / L

        # Pack updated states
        self.states = np.array([
            x_new,
            y_new,
            psi_new,
            psiDot_new,
        ],dtype=float)
