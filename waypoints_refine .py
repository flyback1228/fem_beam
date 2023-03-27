#!/usr/bin/env python

import rospy
from ackermann_planner.srv import *
from geometry_msgs.msg import Pose2D
from ackermann_planner.msg import Pose2DArray
import math
import numpy as np
import matplotlib.pyplot as plt
#from acsr_geometry import ObstacleForce
import shapely
import shapely.ops
from elastica.external_forces import NoForces


from elastica import *
import casadi

class WaypointsRefineSimulator(
    BaseSystemCollection, Constraints, Forcing, Damping,CallBacks
):
    pass

class WaypointsRefineCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        """save the positions of every iteration
        """
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(
                system.position_collection.copy()
            )
            return


class ObstacleForce(NoForces):
    def __init__(self, data, k=2e6):
        """Obstacle force applied to car

        Args:
            data (list like): store the obstacles
            k (double): potential field strength. Defaults to 2e6.
        """
        super().__init__()
        self.polygons=[]
        self.linerings=[]
        self.index_of_polygons=[]
        i=int(0)
        # print(len(data))
        # self.points=[]
        for d in data:
            d = np.reshape(d,(-1,2))
            polygon = shapely.Polygon(d)
            # for i in range(0,len(d)-1):
            #     pt = shapely.Point(d[i])
            #     self.points.append(pt)                
            self.polygons.append(polygon)
            self.linerings.append(polygon.exterior)
            self.index_of_polygons.append(i)
            for inter in polygon.interiors:
                self.linerings.append(inter)
                self.index_of_polygons.append(i)
            i+=1

        self.tree = shapely.STRtree(self.linerings)
        # self.tree = shapely.MultiPoint(self.points)
        self.k = k


    def apply_forces(self, system, time: np.float64 = 0.0):
        tangents = system.tangents[0:2, :].T
        positons = system.position_collection[0:2, :].T

        pts = [shapely.Point(p[0:2]) for p in positons[1:-1]]
        indice = self.tree.nearest(pts)

        for i in range(len(pts)):
            p1,p2 = shapely.ops.nearest_points(pts[i],self.linerings[indice[i]])

            direction = np.array([p1.x-p2.x,p1.y-p2.y,0.0])
            direction = (direction/np.linalg.norm(direction)).reshape(3,)
            # direction = direction*np.array([-tangents[i,1],tangents[i,0],0.0])
            
            polygon_index=self.index_of_polygons[indice[i]]
            if(self.polygons[polygon_index].contains(p1)):
                system.external_forces[:,i]=-1e11*direction
            else:
                distance = shapely.distance(p1,p2)
                force = min(self.k/distance/distance,1e11)
                system.external_forces[:,i]=force*direction

        # for i in range(1,len(positons)-1):            
        #     orig = shapely.Point(positons[i,0:2])
        #     orig,nearest_pt = shapely.ops.nearest_points(orig, self.tree)
        #     distance = shapely.distance(orig,nearest_pt)
        #     direction = np.array([orig.x-nearest_pt.x,orig.y-nearest_pt.y,0.0])
        #     for p in self.polygons:
        #         if(p.contains(orig)):
        #             direction=-direction
        #             direction = (direction/np.linalg.norm(direction)).reshape(3,)
        #             direction = direction*np.array([-tangents[i,1],tangents[i,0],0.0])
        #             # force = min(self.k/distance/distance,1e11)
        #             system.external_forces[:,i]=1e11*direction
        #             break

        #     direction = (direction/np.linalg.norm(direction)).reshape(3,)
        #     direction = direction*np.array([-tangents[i,1],tangents[i,0],0.0])
        #     force = min(self.k/distance/distance,1e11)
        #     system.external_forces[:,i]=force*direction

        # pts = [shapely.Point(p[0:2]) for p in positons]
        # indice = self.tree.nearest(pts)

        # for i in range(1,len(pts)-1):
        #     p1,p2 = shapely.ops.nearest_points(pts[i],self.shapes[indice[i]])
        #     distance = shapely.distance(p1,p2)
        #     direction = np.array([p1.x-p2.x,p1.y-p2.y,0.0])
        #     direction = (direction/np.linalg.norm(direction)).reshape(3,)
        #     #direction = direction*np.array([tangents[i,1],-tangents[i,0],0.0])
        #     direction = direction*np.array([-tangents[i,1],tangents[i,0],0.0])
        #     force = min(self.k/distance/distance,1e10)
        #     system.external_forces[:,i]=force*direction


class ObstacleTree():
    def __init__(self, data):
        """Obstacle force applied to car

        Args:
            data (list like): store the obstacles
            k (double): potential field strength. Defaults to 2e6.
        """

        self.shapes=[]
        for d in data:
            d = np.reshape(d,(-1,2))
            self.shapes.append(shapely.Polygon(d))
        self.tree = shapely.STRtree(self.shapes)
        


    def get_margin(self, positions):
        
        pts = [shapely.Point(p[0:2]) for p in positions]
        indice = self.tree.nearest(pts)
        margin=1000.0
        for i in range(0,len(pts)):
            p1,p2 = shapely.ops.nearest_points(pts[i],self.shapes[indice[i]])
            distance = shapely.distance(p1,p2)
            margin = distance if distance<margin else margin
        return margin
            

class CurvatureTorch(NoForces):
    def __init__(self,k,min_radius):
        """if the curvature of the beam is too great (radius too small), this opposite torch will be applied to the beam

        Args:
            k (_type_): coefficient
            min_radius (_type_): control radius
        """
        self.k = k
        self.max_kappa = 1.0/math.fabs(min_radius)
        self.direction=np.array([1.0, 0.0, 0.0])

    def apply_forces(self, system, time: np.float64 = 0.0):
        kappa = system.kappa[0,:]
        for i in range(0,len(kappa)):
            if math.fabs(kappa[i]<self.max_kappa):
                sign = 1*kappa[i]/math.fabs(kappa[i])
                system.external_torques[:,i] = self.k*kappa[i]*kappa[i]*self.direction*sign

def process(init_state,obstacls,damper,plot = False,interp=False,interp_num=10):
    """_summary_

    Args:
        init_state (_type_): the initial states [x,y,theta] from sst. Note only the theta of the first state is used.
        obstacls (_type_): obstacle list
        plot (bool, optional): can be ignored. Defaults to False.
        interp (bool, optional): can be ignored. Defaults to False.
        interp_num (int, optional): can be ignored. Defaults to 10.

    Returns:
        list of states: the final states
    """
    if interp:
        interp_state = np.zeros([init_state.shape[0]*interp_num,init_state.shape[1]])
        x=range(0,len(init_state))
        xvals = np.linspace(0, len(init_state)-1, interp_num*len(init_state))
        interp_state[:,0] = np.interp(xvals, x, init_state[:,0])
        interp_state[:,1] = np.interp(xvals, x, init_state[:,1])
        if init_state.shape[1]>2:
            interp_state[:,2] = np.interp(xvals, x, init_state[:,2])
        state=interp_state#[0:-1,:]

    else:
        state=init_state

    # n_elem = len(state)-1
    # positions = np.zeros((3,n_elem+1))
    # positions[0:2,:] = state.T[0:2,:]

    n_elem = len(state)-1
    positions = np.zeros((3,n_elem+1))
    positions[0:2,2:] = state.T[0:2,2:]
    positions[0:2,0] = state.T[0:2,0]


    if state.shape[1]>2:
        phi0 = state[0,2]
    else:
        phi0 =math.atan2(state[1,1]-state[0,1],state[1,0]-state[0,0])

    start=np.zeros((3,))
    start[0:2]=state[0,0:2]

    direction = np.array([np.cos(phi0), np.sin(phi0), 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    # d1 = np.linalg.norm(positions[0:2,2]-positions[0:2,0])/2
    # positions[:,1] = positions[:,0]+d1*direction

    p0 = casadi.reshape(casadi.DM(positions[0:2,0]),2,1)
    p2 = casadi.reshape(casadi.DM(positions[0:2,2]),2,1)
    direction1 = casadi.DM([np.cos(phi0), np.sin(phi0)])
    d1 = casadi.norm_2(p2-p0)

    opti = casadi.Opti()
    p1 = opti.variable(2,1)
    d0 = casadi.norm_2(p2-p1)
    d2 = casadi.norm_2(p0-p1)
    opti.subject_to((p1-p0)/d2==direction1)
    #opti.subject_to(d2<d1)
    # opti.subject_to(d0<d1)
    opti.subject_to(d2>0)
    opti.minimize(casadi.dot(d0-d1,d0-d1)+casadi.dot(d1-d2,d1-d2)+casadi.dot(d2-d0,d2-d0))
    opti.set_initial(p1,p0+direction1*d1/2)
    casadi_opts={'print_time':False}
    opopt_opts = {'print_level':0}
    opti.solver('ipopt',casadi_opts,opopt_opts)
    sol = opti.solve()

    print('Insert the direction guide point: {}'.format(sol.value(p1)))
    positions[0:2,1] = sol.value(p1)

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