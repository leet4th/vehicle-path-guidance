import numpy as np

class SteeringControl:

    def __init__(self, dt, gains):

        self.dt = dt

        # Controller gains
        self.gains = gains
        self.Kp_cte = gains['Kp_cte']
        self.Kp_vMag = gains['Kp_vMag']
        self.Kp_cte_vMag = gains['Kp_cte_vMag']
        self.Kp_psi = gains['Kp_psi']
        self.Kd_psi = gains['Kd_psi']
        self.cmd_limit = gains['cmd_limit']

    def update(self, cte, psi_error, psiRate_error, vMag):
        """ Returns steering command angle in radians """

        # Cross track error component
        self.cmd_cte = np.arctan2( self.Kp_cte*cte, (1.0+self.Kp_vMag*vMag) )
        self.cmd_cte *= self.Kp_cte_vMag

        # Proportional term on Heading error
        self.cmd_psi_P = self.Kp_psi * psi_error

        # Derivative Term on Rate Error
        self.cmd_psi_D = self.Kd_psi * psiRate_error

        # Steering command
        cmd_steer = self.cmd_cte + self.cmd_psi_P + self.cmd_psi_D

        # Limit steering command
        self.cmd_steer = np.clip(cmd_steer, self.cmd_limit[0], self.cmd_limit[1])

        return self.cmd_steer

