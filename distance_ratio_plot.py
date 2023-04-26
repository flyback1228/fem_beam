import numpy as np
import os
import casadi as ca
from ObstacleForce import ObstacleForce
import matplotlib.pyplot as plt
from Obstacle import Obstacle
import fiona
from shapely.geometry import shape



script_dir = os.path.dirname(__file__)
obstacles=[]
obstacle_file = os.path.join(script_dir, 'data/fem_refine/extended_polygon.shp')
with fiona.open(obstacle_file) as shapefile:
    for record in shapefile:
        geometry = shape(record['geometry'])
        x5,y5 = geometry.exterior.xy
        obstacles.append(np.vstack([x5,y5]).transpose())
        
sst_file = os.path.join(script_dir, 'data/fem_refine/sst_data.txt')
sst_state = np.array(ca.DM.from_file(sst_file))

ces_file = os.path.join(script_dir, 'result/distance_ratio/ces.txt')
ces = np.array(ca.DM.from_file(ces_file))

no_ratio_file = os.path.join(script_dir, 'result/distance_ratio/ratio_0.0.txt')
no_ratio = np.array(ca.DM.from_file(no_ratio_file))

ratio_010_file = os.path.join(script_dir, 'result/distance_ratio/ratio_0.1.txt')
ratio_010 = np.array(ca.DM.from_file(ratio_010_file))

ratio_005_file = os.path.join(script_dir, 'result/distance_ratio/ratio_0.05.txt')
ratio_005 = np.array(ca.DM.from_file(ratio_005_file))

ratio_020_file = os.path.join(script_dir, 'result/distance_ratio/ratio_0.2.txt')
ratio_020 = np.array(ca.DM.from_file(ratio_020_file))

ax = plt.subplot(1,1,1)
obstacle_force = ObstacleForce(obstacles)
obstacle_force.plot_obstacles(ax)
# ax.plot(sst_state[0:3,0],sst_state[0:3,1],label='{}'.format(0))
ax.arrow(0,0,2*np.cos(sst_state[0,2]),2*np.sin(sst_state[0,2]),head_width = 0.5, head_length = 1,label='Initial Orientation')
ax.plot(sst_state[:,0],sst_state[:,1],'-.r',label='SST',linewidth=2)
ax.plot(ces[:,0],ces[:,1],'--m',label='CES',linewidth=2)
ax.plot(ratio_005[:,0],ratio_005[:,1],'-b',label='Stretching Ratio=0.05',linewidth=2)
ax.plot(ratio_010[:,0],ratio_010[:,1],'-y',label='Stretching Ratio=0.10',linewidth=2)
ax.plot(ratio_020[:,0],ratio_020[:,1],'-k',label='Stretching Ratio=0.20',linewidth=2)
ax.plot([0],[0],'-g',label='Obstacles')

plt.legend()
plt.show()


