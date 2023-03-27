#!/usr/bin/env python

from ObstacleForce import *
import casadi
import numpy as np


def process(init_state,obstacls,damper,plot = False):
    """_summary_

    Args:
        init_state (list): the initial states [x,y,theta] from sst. Note only the theta of the first state is used.
        obstacls (list of polygon): obstacle list
        plot (bool, optional): can be ignored. Defaults to False.
        

    Returns:
        list of states: the final states
    """
    state=init_state

    # n_elem = len(state)-1
    # positions = np.zeros((3,n_elem+1))
    # positions[0:2,:] = state.T[0:2,:]

    n_elem = len(state)-1
    positions = np.zeros((n_elem+1,2))
    positions[2:,:] = state[2:,0:2]
    positions[0,:] = state[0,0:2]


    if len(state[0]==2):
        phi0 =np.arctan2(state[1,1]-state[0,1],state[1,0]-state[0,0])
        positions[1,:]=state[1,0:2]
        
    else:
        phi0 = state[0,2]

        # start=np.zeros((3,))
        # start[0:2]=state[0,0:2]

        direction = np.array([np.cos(phi0), np.sin(phi0)])

        p0 = casadi.DM(positions[0,:])
        p2 = casadi.DM(positions[2,:])
        direction_dm = casadi.DM(direction)
        d1 = casadi.norm_2(p2-p0)

        opti = casadi.Opti()
        p1 = opti.variable(2,1)
        d0 = casadi.norm_2(p2-p1)
        d2 = casadi.norm_2(p0-p1)
        opti.subject_to((p1-p0)/d2==direction_dm)
        #opti.subject_to(d2<d1)
        # opti.subject_to(d0<d1)
        opti.subject_to(d2>0)
        opti.minimize(casadi.dot(d0-d1,d0-d1)+casadi.dot(d1-d2,d1-d2)+casadi.dot(d2-d0,d2-d0))
        opti.set_initial(p1,p0+direction_dm*d1/2)
        casadi_opts={'print_time':False}
        opopt_opts = {'print_level':0}
        opti.solver('ipopt',casadi_opts,opopt_opts)
        sol = opti.solve()

        print('Insert the direction guide point: {}'.format(sol.value(p1)))
        positions[1,:] = sol.value(p1)

    # print(sol.value(casadi.dot(d0-d1,d0-d1)+casadi.dot(d1-d2,d1-d2)+casadi.dot(d2-d0,d2-d0)))
    # print(positions[:,0:3])

    final_time = 1
    dt = 0.01

    base_length = 0.1
    coeff = np.linspace(1,n_elem*1.2,n_elem)
    base_radius = 1*n_elem*np.ones((n_elem,))/coeff

    # base_radius[0:3]=200*base_radius[0:3]

    """the following four varibles affect the final shape.
        Basically, greater values cause greater mass and stiffness matrix, thus the reshape velocity will be increased
    """
    density = 1e9
    youngs_modulus = 2e10
    poisson_ratio = 0.03
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    shear_modulus = youngs_modulus

    plot_history=[np.copy(positions)]
    lengths = np.linalg.norm(positions[0:2,1:]-positions[0:2,0:-1],axis=0)
    rest_lengths = 0.8*np.linalg.norm(positions[0:2,-1]-positions[0:2,0])/(positions.shape[1])

    #total_its, total time and total steps affects the final shape
    total_its = 40
    plot_lines = 4
    save_indice = int(total_its/plot_lines)
    total_steps = 400
    total_time = 4.0
    for i in range(0,total_its+1):
        print('iter = {}/{}'.format(i,total_its))
        # lengths = np.linalg.norm(positions[0:2,1:]-positions[0:2,0:-1],axis=0)

        waypoints_sim = WaypointsRefineSimulator()
        waypoints_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            0.0,  # internal damping constant, deprecated in v0.3.0
            youngs_modulus,
            shear_modulus=shear_modulus,
            position=positions,
        )

        #set the defaut length
        waypoints_rod.rest_lengths = rest_lengths

        #set the constraint of the ends. the initial end is fixed and the last end is pined
        waypoints_sim.append(waypoints_rod)
        waypoints_sim.constrain(waypoints_rod).using(
            FixedConstraint,
            constrained_position_idx=(0,-1),
            constrained_director_idx=(0,),
        )

        #set the damp of system. Note damping_constant affects the convergent speed
        waypoints_sim.dampen(waypoints_rod).using(
            AnalyticalLinearDamper,
            # damping_constant=15,
            damping_constant = damper,
            time_step=0.1,
        )

        #applying obstacle force
        waypoints_sim.add_forcing_to(waypoints_rod).using(
            ObstacleForce,
            data=obstacls,
            k=5e8
        )

        # #applying minimal curvature torch
        # waypoints_sim.add_forcing_to(waypoints_rod).using(
        #     CurvatureTorch,
        #     k=1e6,
        #     min_radius = 0.35/math.tan(30.0/180*math.pi)
        # )

        recorded_history = defaultdict(list)

        waypoints_sim.collect_diagnostics(waypoints_rod).using(
            WaypointsRefineCallBack, step_skip=1, callback_params=recorded_history
        )
        waypoints_sim.finalize()
        # print(waypoints_rod.tangents)
        timestepper = PositionVerlet()
        integrate(timestepper, waypoints_sim, total_time, total_steps,progress_bar=True)

        recorded_history["position"].append(waypoints_rod.position_collection.copy())
        recorded_history["direction"].append(waypoints_rod.tangents.copy())
        recorded_history["director"].append(waypoints_rod.director_collection.copy())
        recorded_history["kappa"].append(waypoints_rod.kappa[0,:].copy())

        positions = recorded_history["position"][-1]

        if plot and i%save_indice==0:
            plot_history.append(positions)

    directions = recorded_history["direction"][-1]
    directors = recorded_history["director"][-1]
    kappa = recorded_history["kappa"][-1]

    lengths = np.linalg.norm(positions[0:2,1:]-positions[0:2,0:-1],axis=0)

    # print(kappa)
    if plot:
        shapes=[]
        for d in obstacls:
            d = np.reshape(d,(-1,2))
            shapes.append(shapely.Polygon(d))
        for poly in shapes:
            x,y = poly.exterior.xy
            plt.plot(x,y,'-g')

        for index,position in enumerate(plot_history):
            plt.plot(position[0,:],position[1,:],label="{}".format(index))
        # plt.plot(plot_history[0][0,:],plot_history[0][1,:],label="{}".format(0),marker="o")
        # plt.plot(plot_history[-1][0,:],plot_history[-1][1,:],label="{}".format(-1),marker="o")

        # for i in range(0,len(directions.T)):
        #     plt.arrow(plot_history[-1][0,i],plot_history[-1][1,i],directions[0,i],directions[1,i])


        plt.legend()
        
        plt.savefig("/home/acsr/Documents/racecar_planner_temp/refine.svg",dpi=500)
        # plt.show()
        #

    return positions


def handle_request(req):
    obs_list=[]
    for poly in req.obstacles.polygons:
        vec =[]
        for pt in poly.points:
            vec.append(pt.x)
            vec.append(pt.y)
        obs_list.append(np.array(vec))

    print("poses")
    print(req.sst_data.poses)
    sst_data=[]
    for sst in req.sst_data.poses:
        sst_data.append(sst.x)
        sst_data.append(sst.y)
        sst_data.append(sst.theta)

    sst_data = np.array(sst_data).reshape(-1,3)

    process_data = process(sst_data,obs_list,damper=req.damper,plot=True)
    # print(process_data)

    obs = ObstacleTree(data=obs_list)
    margin = obs.get_margin(process_data.T)
    # margin=0.3 if margin>0.3 else margin

    refine_data =[]
    for i in range(len(process_data.T)):
        pose=Pose2D()
        pose.x = process_data[0,i]
        pose.y=process_data[1,i]
        refine_data.append(pose)
    
    refined_pose = Pose2DArray()
    refined_pose.poses = refine_data
    
    return WaypointsResponse(refined_data=refined_pose,margin=margin)
    

def waypoints_refine_server():
    rospy.init_node('waypoints_refine_server')
    s = rospy.Service('/move_base/CarSSTPlanner/waypoints_refine', Waypoints, handle_request)
    print("Ready to process waypoints.")
    rospy.spin()

if __name__ == "__main__":
    waypoints_refine_server()