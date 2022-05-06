import numpy as np
import matplotlib.pyplot as plt

from Guidance import GuidanceModeHandler
from Steering import SteeringControl
from VehicleModels import RearWheelBicycleModel

def run_sim(tStart, tEnd, dt, guidHandler, ctrl, sim):

    # Setup sim time
    time = np.arange(tStart,tEnd/dt+dt)*dt

    # Constant vMag
    vMag = guidHandler.vMag

    # Run Sim
    output = {}
    output['states'] = []
    output['inputs'] = []
    output['path_dist'] = []
    output['cte'] = []
    output['heading_error'] = []
    output['headingRate_error'] = []
    output['rear'] = []
    output['front'] = []
    for k,tk in enumerate(time):

        # States [x, y, psi, psiDot]
        psi = sim.states[2]
        psiDot = sim.states[3]

        # Guidance
        point = tuple(sim.front)
        #point = tuple(sim.rear)
        progress, cte, psi_error, psiDot_error = guidHandler.update(point, psi, psiDot)

        # Lateral Control
        steer_angle = ctrl.update(cte, psi_error, psiDot_error, vMag)

        # Update sim
        sim.update(steer_angle, vMag)

        # Save output
        output['inputs'].append([steer_angle, vMag])
        output['states'].append(sim.states)
        output['rear'].append(sim.rear)
        output['front'].append(sim.front)
        output['path_dist'].append(progress)
        output['cte'].append(cte)
        output['heading_error'].append(psi_error)
        output['headingRate_error'].append(psiDot_error)

    # Convert to numpy
    for key,val in output.items():
        output[key] = np.array(val)

    output['time'] = time


    return output

if __name__== "__main__":

    wantPlot = True

    sweep = {}
    sweep_Kp_cte = np.linspace(0.1,3,5)
    sweep_Kp_cte = [18.75]

    for Kp_cte in sweep_Kp_cte:

        # Time
        tStart = 0.0
        tEnd = 100.0
        dt = 0.01

        # Setup sim
        initial_state = np.array([
            10.0,     # x
            30.0,     # y
            90*np.pi/180, # psi
            0.0,     # psiDot
        ],dtype=float)
        params = {
            'dt':dt,
            'L':2.5,
            'max_steer_angle_deg': 30.0,
        }
        sim = RearWheelBicycleModel(initial_state=initial_state, params=params)

        # Constant speed control
        vMag_kph = 20.0
        vMag = vMag_kph / 3.6

        print('vMag_kph ', vMag_kph)
        print('vMag     ', vMag)


        # Setup guidance
        waypoint_A = (1,10)
        waypoint_B = (50,50)
        d = 4
        R = 15
        spline_dt = 0.0001
        window_size = 0.25
        guidHandler = GuidanceModeHandler(vMag, waypoint_A, waypoint_B, d, R, spline_dt, window_size)

        # Setup Control
        gains = {}
        gains['Kp_cte'] = 30.0 #Kp_cte*5
        gains['Kp_vMag'] = 1.0
        gains['Kp_cte_vMag'] = 1.0
        gains['Kp_psi'] = 1.4
        gains['Kd_psi'] = 0.025

        #gains['Kp_cte'] = 18.5
        #gains['Kp_vMag'] = 1.0
        #gains['Kp_cte_vMag'] = 1.0
        #gains['Kp_psi'] = 1.4
        #gains['Kd_psi'] = 0.025

        max_cmd = params['max_steer_angle_deg'] * np.pi/180
        gains['cmd_limit'] = np.array([-max_cmd, max_cmd])

        ctrl = SteeringControl(dt, gains=gains)

        # Run sim
        output = run_sim(tStart, tEnd, dt, guidHandler, ctrl, sim)


        cte_rms = np.sqrt( np.mean( output['cte']**2 ) )
        for key,val in gains.items():
            print(f'{key} = {val}')
        print(f'cte_rms = {cte_rms}')

        if wantPlot:
            time = output['time']
            full_path = np.vstack(guidHandler.planned_path)

            fig,ax = plt.subplots(figsize=(10,8))
            ax.set_title('World Frame')
            #ax.plot(guidTurn.path[:,0], guidTurn.path[:,1],'grey',alpha=0.6,lw=10,zorder=-1,label='path')
            ax.plot(full_path[:,0], full_path[:,1],'grey',alpha=0.6,lw=10,zorder=-1,label='path')
            #ax.plot(guidTurn.waypoints_x, guidTurn.waypoints_y, 'co', label='waypoints')
            ax.plot(output['rear'][:,0], output['rear'][:,1],label='rear')
            ax.plot(output['front'][:,0], output['front'][:,1],label='front')
            ax.plot(waypoint_A[0], waypoint_A[1], 'bo', ms = 5,label='Waypoint A')
            ax.plot(waypoint_B[0], waypoint_B[1], 'ro', ms = 5,label='Waypoint B')
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Y(m)')
            ax.legend()
            ax.grid()

            fig,ax = plt.subplots(5,1,sharex=True,figsize=(6,8))
            ax[0].plot(time, output['path_dist'])
            ax[0].axhline(0,ls='--',color='k')
            ax[0].axhline(1,ls='--',color='k')
            ax[0].set_ylabel('progress')
            ax[1].plot(time, output['inputs'][:,0]*180/np.pi)
            ax[1].set_ylabel('steer angle (deg)')
            ax[1].axhline(-params['max_steer_angle_deg'],ls='--',color='k')
            ax[1].axhline(params['max_steer_angle_deg'],ls='--',color='k')
            ax[2].plot(time, output['cte'])
            ax[2].axhline(0,ls='--',color='k')
            ax[2].set_ylabel('cte (m)')
            #ax[2].set_ylim([-1.2,1.2])
            ax[2].set_ylim([-0.12,0.12])
            ax[3].plot(time, output['heading_error']*180/np.pi)
            ax[3].set_ylabel('psi error (deg)')
            ax[4].plot(time, output['headingRate_error']*180/np.pi)
            ax[4].set_ylabel('psiDot error (deg/s)')
            ax[4].set_xlabel('time (s)')
            for i in range(5):
                ax[i].grid()

            fig,ax = plt.subplots(sharex=True)
            ax.plot(time, output['inputs'][:,0]*180/np.pi)
            ax.set_ylabel('steering angle (deg)')
            ax.set_xlabel('time (s)')
            ax.axhline(-params['max_steer_angle_deg'],ls='--',color='k')
            ax.axhline(params['max_steer_angle_deg'],ls='--',color='k')
            ax.grid()

            fig,ax = plt.subplots(1,1,sharex=True)
            ax.plot(output['cte'], output['heading_error']*180/np.pi)
            ax.set_ylabel('psi error (deg)')
            ax.set_xlabel('cte (m)')
            ax.grid()

            plt.show()
