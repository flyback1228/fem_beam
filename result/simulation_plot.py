import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from Obstacle import Obstacle
from spline import direct_spline,parametric_function
from shapely import Point,LineString
import shapely
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import casadi as ca
# from ObstacleForce import ObstacleForce
import matplotlib.pyplot as plt
# from Obstacle import Obstacle
import fiona
from shapely.geometry import shape
import csv
from track import Track
from optimizer import optimize

script_dir = os.path.dirname(__file__)
obstacles=[]
obstacle_file = os.path.join(script_dir, '../data/ces_compare/extended_polygon.shp')
with fiona.open(obstacle_file) as shapefile:
    for record in shapefile:
        geometry = shape(record['geometry'])
        x5,y5 = geometry.exterior.xy
        obstacles.append(np.vstack([x5,y5]).transpose())
        
sst_file = os.path.join(script_dir, '../data/ces_compare/rrt_data.txt')
# sst_state = np.array(ca.DM.from_file(sst_file))
with open(sst_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    for row in csv_reader:
        data.append([float(row[0]),float(row[1])])


sst_state = np.zeros((2*len(data)-1,3))
sst_state[0::2,0:2] = np.array(data)
sst_state[1::2,0:2] = (sst_state[0:-1:2,0:2]+sst_state[2::2,0:2])/2

phi0 = -170/180.0*np.pi
sst_state[0,2]=phi0
    

ces_file = os.path.join(script_dir, '../result/simulation/ces.txt')
ces = np.array(ca.DM.from_file(ces_file))

result_file = os.path.join(script_dir, '../result/simulation/it_80.txt')
result = np.array(ca.DM.from_file(result_file))

fig,ax = plt.subplots(1,1)
obstacle_force = Obstacle(obstacles)
obstacle_force.plot_obstacles(ax)
# ax.plot(sst_state[0:3,0],sst_state[0:3,1],label='{}'.format(0))
ax.arrow(sst_state[0,0],sst_state[0,1],5*np.cos(sst_state[0,2]),5*np.sin(sst_state[0,2]),head_width = 2, head_length = 2,label='Initial Orientation')
ax.plot(sst_state[:,0],sst_state[:,1],'-.c',label='SST',linewidth=2)
ax.plot(ces[:,0],ces[:,1],'--m',label='CES',linewidth=2)
ax.plot(result[:,0],result[:,1],'-y',label='BFS',linewidth=2)
ax.plot([0],[0],'-g')


#plot track
p1_1,p2_1 = direct_spline(result,[np.cos(sst_state[0,2]),np.sin(sst_state[0,2])])
f = parametric_function(result,p1_1,p2_1)

tau = ca.MX.sym('t')
pt_t = f(tau)
n = ca.MX.sym("n")
jac = ca.jacobian(pt_t,tau)
f_tangent = ca.Function('tangent',[tau],[jac])

theta = np.pi/2
rot_mat = ca.DM([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);

# points = [Point(waypoint) for waypoint in result]

# obstacle_force.get_neareset_points(result)

d_forward=[]
inner_forward=[]
outer_forward=[]
list_d_in_forward=[]
list_d_out_forward=[]

n_max = 2.25
n_step = n_max/5

N = len(result)
for i in range(N):
    
    pt0 = Point(result[i,:])
    t = i
    if i>0:
        t=i-0.000001
    
    pt1 = Point(result[i,:]+ca.mtimes(rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*n_max)
    pt2 = Point(result[i,:]+ca.mtimes(-rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*n_max)
    line1 = LineString([pt0,pt1])
    line2 = LineString([pt0,pt2])
    d_in = n_max
    d_out = n_max
    
    if len(inner_forward)>0:
        d_in = min(list_d_in_forward[-1]+n_step,n_max)    
        d_out = min(list_d_out_forward[-1]+n_step,n_max)    
    
        
    for l in obstacle_force.polygons:
        intersect1 = line1.intersection(l)
        if intersect1:
            d = shapely.distance(pt0,intersect1)
            d_in = min(d_in,d)    
            
        intersect2 = line2.intersection(l)
        if intersect2:
            d = shapely.distance(pt0,intersect2)
            d_out = min(d_out,d)
            
    list_d_in_forward.append(d_in)
    list_d_out_forward.append(d_out)

    pt1 = Point( result[i,:]+ca.mtimes(rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*d_in)
    inner_forward.append([pt1.x,pt1.y]) 
    pt2 = Point( result[i,:]+ca.mtimes(-rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*d_out)
    outer_forward.append([pt2.x,pt2.y])   

inner_backward=[]
outer_backward=[]
list_d_in_backward=[]
list_d_out_backward=[]

for i in range(N-1,-1,-1):
    
    pt0 = Point(result[i,:])
    t = i
    if i>0:
        t=i-0.000001
    
    pt1 = Point(result[i,:]+ca.mtimes(rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*n_max)
    pt2 = Point(result[i,:]+ca.mtimes(-rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*n_max)
    line1 = LineString([pt0,pt1])
    line2 = LineString([pt0,pt2])
    d_in = n_max
    d_out = n_max
    
    if len(inner_backward)>0:
        d_in = min(list_d_in_backward[-1]+n_step,n_max)    
        d_out = min(list_d_out_backward[-1]+n_step,n_max)    
    
        
    for l in obstacle_force.polygons:
        intersect1 = line1.intersection(l)
        if intersect1:
            d = shapely.distance(pt0,intersect1)
            d_in = min(d_in,d)    
            
        intersect2 = line2.intersection(l)
        if intersect2:
            d = shapely.distance(pt0,intersect2)
            d_out = min(d_out,d)
            
    list_d_in_backward.append(d_in)
    list_d_out_backward.append(d_out)

    pt1 = Point( result[i,:]+ca.mtimes(rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*d_in)
    inner_backward.append([pt1.x,pt1.y]) 
    pt2 = Point( result[i,:]+ca.mtimes(-rot_mat,f_tangent(t)/ca.norm_2(f_tangent(t)))*d_out)
    outer_backward.append([pt2.x,pt2.y])       

inner_waypoints=[]
outer_waypoints=[]
for i in range(N):
    if list_d_in_backward[N-1-i]>list_d_in_forward[i]:
        inner_waypoints.append(inner_forward[i])
    else:
        inner_waypoints.append(inner_backward[N-1-i])
        
    if list_d_out_backward[N-1-i]>list_d_out_forward[i]:
        outer_waypoints.append(outer_forward[i])
    else:
        outer_waypoints.append(outer_backward[N-1-i])
     
inner_waypoints = np.array(inner_waypoints)
outer_waypoints = np.array(outer_waypoints)

# inner_p1_1,inner_p2_1 = direct_spline(inner_waypoints,[1.0,0.5])
# f_in = parametric_function(inner_waypoints,inner_p1_1,inner_p2_1)

# outer_p1_1,outer_p2_1 = direct_spline(outer_waypoints,[1.0,0.5])
# f_out = parametric_function(outer_waypoints,outer_p1_1,outer_p2_1)

t = ca.linspace(0,N-1-0.01,100).T
l = ca.reshape(f(t),(2,-1)).T
# l_in = ca.reshape(f_in(t),(2,-1)).T
# l_out = ca.reshape(f_out(t),(2,-1)).T


track = Track(result,[np.cos(phi0),np.sin(phi0)])
t_val,x_val,u_val = optimize(track,n_max*2)

xy = track.f_tn_to_xy(ca.DM(x_val[0,:]).T,ca.DM(x_val[1,:]).T)

# fig, ax = plt.subplots(1,1)
ax.plot(l[:,0],l[:,1],'-r',label='Track Centerline')
# plt.plot(l_in[:,0],l_in[:,1],'--b',label='left boundary')
# plt.plot(l_out[:,0],l_out[:,1],'--g',label='right boundary')

ax.plot(inner_waypoints[:,0],inner_waypoints[:,1],'--b',label='Track Boundary')
ax.plot(outer_waypoints[:,0],outer_waypoints[:,1],'--b')

xy = np.transpose(xy).reshape(-1,1,2)
segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
vx = x_val[3,:]

norm = plt.Normalize(vx.min(), vx.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(vx)
lc.set_linewidth(3)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

# print(segments)
# plt.plot(np.reshape(xy[0,:],(-1,)),np.reshape(xy[1,:],(-1,)),'-k',linewidth = 3,label='Optimized Trajectory')

plt.legend(fontsize="12")
plt.show()


