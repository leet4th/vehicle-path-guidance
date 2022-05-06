
import numpy as np
from scipy import interpolate

class GuidanceLine:
    def __init__(self, waypoint_A, waypoint_B):

        # Set waypoints
        self.set_waypoints(waypoint_A, waypoint_B)

    def set_waypoints(self, waypoint_A, waypoint_B):
        """Initialize guidance line

        waypoints A and B defined in world/local coordinate frame

        """
        # Store waypoints
        self.waypoint_A = np.asarray(waypoint_A,dtype=float)
        self.waypoint_B = np.asarray(waypoint_B,dtype=float)
        self.path = np.array([waypoint_A, waypoint_B])

        # Path vector (Straight line path)
        self.path_vec = self.waypoint_B - self.waypoint_A
        self.path_length = np.sqrt(np.dot(self.path_vec, self.path_vec))

        # Path heading
        self.path_heading = np.arctan2(self.path_vec[1], self.path_vec[0])

        # Define Guidance frame (G) DCM
        ch = np.cos(self.path_heading)
        sh = np.sin(self.path_heading)
        self.C_toGfromL = np.array([
            [ ch, sh],
            [-sh, ch],
        ],dtype=float)

    def update(self, point, heading, headingRate):
        """Returns the perpendicular distance between point and line"""

        # Point in world/local frame
        point = np.asarray(point)

        # Transform point to Guidance frame (G)
        point = point - self.waypoint_A
        point_G = np.dot(self.C_toGfromL, point)

        # Progress (normalized distance along path)
        progress = point_G[0] / self.path_length

        # Cross track error (Perpendicular distance from point to line)
        cte = -point_G[1]

        # Heading error
        heading_error = self.path_heading - heading
        if heading_error > np.pi:
            heading_error -= 2*np.pi
        elif heading_error < -np.pi:
            heading_error += 2*np.pi

        # Heading Rate error
        headingRate_error = -headingRate

        return progress, cte, heading_error, headingRate_error

class GuidanceUTurn:
    def __init__(self, vMag, d, R, loc, heading, spline_dt, window_size):

        if R <= 0:
            raise ValueError("R must be greater than 0")

        if d <= 0:
            raise ValueError("d must be greater than 0")

        # Store inputs
        self.d = d
        self.R = R
        self.vMag = vMag
        self.loc = np.asarray(loc)
        self.path_heading = heading
        self.spline_dt = spline_dt
        self.window_size = window_size

        # Setup U-Turn path and generate waypoints
        self.set_path()
        self.set_waypoints()

        self.kTarget_prev = 0


    def set_path(self):
        """ U-Turn Path consisting of 6 segments """

        # Define Guidance frame (G) DCM
        ch = np.cos(self.path_heading)
        sh = np.sin(self.path_heading)
        self.C_toGfromL = np.array([
            [ ch, sh],
            [-sh, ch],
        ],dtype=float)

        # Setup path
        theta = np.pi/3
        sth = np.sin(theta)

        # Path segments in Guidance frame
        path_segments_G = []

        # Path 1
        x = np.linspace(0, self.d, 1000)
        y = 0*x

        path1 = np.array([x,y]).T
        self.path1_start = 0.0
        self.path1_end = self.d

        # Path 2
        a2 = 0
        b2 = self.R*sth
        offset2 = np.array([self.d, 0])

        x = np.linspace(a2,b2)
        y = -np.sqrt(self.R**2 - x**2) + self.R

        path2 = np.array([x,y]).T + offset2
        self.path2_start = self.path1_end
        self.path2_end = self.path2_start + self.R*theta

        # Path 3
        a3 = -b2
        b3 = self.R
        offset3 = np.array([2*self.R*sth + self.d, 0])

        x = np.linspace(a3, b3, 1000)
        y = np.sqrt(self.R**2 - x**2)
        path3 = np.array([x,y]).T + offset3
        self.path3_start = self.path2_end
        self.path3_end = self.path3_start + self.R*(np.pi-theta/2)

        # Path 4
        path4 = path3[::-1] * np.array([1,-1])
        self.path4_start = self.path3_end
        self.path4_end = self.path4_start + self.R*(np.pi-theta/2)

        # Path 5
        path5 = path2[::-1] * np.array([1,-1])
        self.path5_start = self.path4_end
        self.path5_end = self.path5_start + self.R*theta

        # Path 6
        path6 = path1[::-1]
        self.path6_start = self.path5_end
        self.path6_end = self.path6_start + self.d

        path_segments_G = [
                path1[:-1],
                path2[:-1],
                path3[:-1],
                path4[:-1],
                path5[:-1],
                path6[:-1],
        ]

        # Transform each segment to world frame from guidance frame
        self.path_segments = [np.dot(self.C_toGfromL.T, pk.T).T + self.loc for pk in path_segments_G]

        # Transform path to L from G
        self.path = np.vstack(self.path_segments)

        # Total Travel
        self.travel = self.path6_end

    def set_waypoints(self):
        """ Set waypoints for path as function of progress along path

        waypoints = {x,y,psi,psiDot}

        """

        # Cumulative distance traveled
        dx = np.diff(self.path[:,0])
        dy = np.diff(self.path[:,1])
        ds = np.sqrt(dx**2 + dy**2)
        self.s  = np.hstack((0, np.cumsum(ds)))


        # Spline independent variable
        spline_t = self.s/self.s[-1]

        # Fit cubic splines
        self.Xfit = interpolate.CubicSpline(spline_t,self.path[:,0],extrapolate=False)
        self.Yfit = interpolate.CubicSpline(spline_t,self.path[:,1],extrapolate=False)

        # Calc waypoints from splines
        # Independent variable
        self.spline_t = np.arange(0, 1+self.spline_dt, self.spline_dt)
        self.spline_t[-1] = 1.0

        # Position waypoints
        self.waypoints_x = self.Xfit(self.spline_t)
        self.waypoints_y = self.Yfit(self.spline_t)

        # Psi waypoints
        dx = self.Xfit.derivative(1)(self.spline_t)
        dy = self.Yfit.derivative(1)(self.spline_t)
        self.waypoints_heading = np.arctan2(dy,dx)

        # PsiDot waypoints
        k_path6 = self.spline_t >= self.path6_start
        k_path5 = np.logical_and(self.spline_t >= self.path5_start, self.spline_t < self.path6_start)
        k_path4 = np.logical_and(self.spline_t >= self.path4_start, self.spline_t < self.path5_start)
        k_path3 = np.logical_and(self.spline_t >= self.path3_start, self.spline_t < self.path4_start)
        k_path2 = np.logical_and(self.spline_t >= self.path2_start, self.spline_t < self.path3_start)
        k_path1 = self.spline_t < self.path2_start

        psiDot = np.zeros_like(self.spline_t)
        # On Path6 (Straight)
        psiDot[k_path6] = 0.0
        # On Path5 (Turn Left)
        psiDot[k_path5] = self.vMag/self.R
        # On Path4 (Turn Right)
        psiDot[k_path4] = -self.vMag/self.R
        # On Path3 (Turn Right)
        psiDot[k_path3] = -self.vMag/self.R
        # On Path2 (Turn Left)
        psiDot[k_path2] = self.vMag/self.R
        # On Path1 (Straight)
        psiDot[k_path1] = 0.0

        self.waypoints_headingRate = psiDot

        # Setup window
        self.n_points = len(self.spline_t)
        self.window = np.array([False]*self.n_points)
        self.window_width = int(self.window_size/self.spline_dt)


    def update(self, point, heading, headingRate):

        point = np.asarray(point)

        # Reset window
        self.window.fill(False)

        # Indices for window, with previous point as mid point of window
        kStart = self.kTarget_prev - int(self.window_width/2)
        if kStart < 0:
            kStart = 0

        kEnd = kStart + self.window_width
        if kEnd > self.n_points:
            kEnd = self.n_points
            kStart = kEnd - self.window_width

        # Set window
        self.window[kStart:kEnd] = True

        # Find index of closest point within window
        dx = point[0] - self.waypoints_x[self.window]
        dy = point[1] - self.waypoints_y[self.window]
        dist = np.sqrt(dx**2 + dy**2)
        kTarget_window = np.argmin(dist)

        # Adjust for use with full array
        kTarget = kTarget_window + kStart

        # Dont look back at previous waypoints
        if kTarget < self.kTarget_prev:
            kTarget = self.kTarget_prev

        # Progress
        progress = self.spline_t[kTarget]

        # Heading and position error
        target_heading = self.waypoints_heading[kTarget]
        target_err = point - np.array([self.waypoints_x[kTarget],self.waypoints_y[kTarget]])

        # DCM from world/local frame to target waypoint
        ch = np.cos(target_heading)
        sh = np.sin(target_heading)
        C_toTargetfromL = np.array([
            [ ch, sh],
            [-sh, ch],
        ],dtype=float)

        # Error to waypoint in target waypoint frame
        err_target_waypointFrame = np.dot(C_toTargetfromL,target_err)
        cte = -err_target_waypointFrame[1]

        # Heading error and handle angle wrapping
        heading_error = target_heading - heading
        if heading_error > np.pi:
            heading_error -= 2*np.pi
        elif heading_error < -np.pi:
            heading_error += 2*np.pi

        # Target Heading Rate for current waypoint
        target_headingRate = self.waypoints_headingRate[kTarget]

        # Heading rate error
        headingRate_error = target_headingRate - headingRate

        self.kTarget_prev = kTarget

        return progress, cte, heading_error, headingRate_error

class GuidanceModeHandler:

    def __init__(self, vMag, waypoint_A, waypoint_B, d, R, spline_dt, window_size):

        # Current mode: 'line' or 'turn'
        self.mode = 'line'
        self.mode_switch = False

        # Track current turn number
        self.n_turns = 0

        # Store vMag
        self.vMag = vMag

        # Setup guidance line
        self.waypoint_A = waypoint_A
        self.waypoint_B = waypoint_B

        self.reset_guidance_line(self.waypoint_A, self.waypoint_B)

        # Setup guidance u-turn
        self.d = d
        self.R = R
        self.spline_dt = spline_dt
        self.window_size = window_size

        loc = self.guidLine.waypoint_B
        heading = self.guidLine.path_heading

        self.reset_guidance_turn(
                self.vMag,
                self.d,
                self.R,
                loc,
                heading,
                self.spline_dt,
                self.window_size
        )

        self.planned_path = []
        self.planned_path.append( self.guidLine.path )

    def reset_guidance_line(self, waypoint_A, waypoint_B):
        """ Resets the guidance line for new inputs """
        self.guidLine = GuidanceLine(waypoint_A, waypoint_B)

    def reset_guidance_turn(self, vMag, d,R,loc,heading,spline_dt,window_size):
        """ Resets the guidance turn for new inputs """
        self.guidTurn = GuidanceUTurn(vMag,d,R,loc,heading,spline_dt,window_size)

    def update(self, point, heading, headingRate):

        self.mode_switch = False

        if self.mode == 'line':
            progress, cte, heading_error, headingRate_error = self.guidLine.update(point, heading, headingRate)

            if progress >= 0.99:
                loc = self.guidLine.waypoint_B
                heading = self.guidLine.path_heading

                self.reset_guidance_turn(
                        self.vMag,
                        self.d,
                        self.R,
                        loc,
                        heading,
                        self.spline_dt,
                        self.window_size
                )
                self.planned_path.append( self.guidTurn.path )

                self.mode = 'turn'
                self.n_turns += 1
                self.mode_switch = True

        elif self.mode == 'turn':
            progress, cte, heading_error, headingRate_error = self.guidTurn.update(point, heading, headingRate)

            if progress >= 0.99:

                if (self.n_turns % 2) == 0:
                    # Even - original direction
                    waypoint_A = self.waypoint_A
                    waypoint_B = self.waypoint_B
                else:
                    # odd  - reverse direction
                    waypoint_A = self.waypoint_B
                    waypoint_B = self.waypoint_A

                self.reset_guidance_line(waypoint_A, waypoint_B)
                self.planned_path.append( self.guidLine.path )

                self.mode = 'line'
                self.mode_switch = True

        else:
            raise ValueError(f"mode {self.mode} not supported")

        return progress, cte, heading_error, headingRate_error

