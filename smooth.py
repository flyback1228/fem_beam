import fiona
from shapely.geometry import shape
import numpy as np
import cupy as cp
import os
import casadi as ca
from ObstacleForce import ObstacleForce
from Node import Node
from Element import Element, ElementCupy
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import time
from Obstacle import Obstacle

def modeling(obstacles,original_data,distance_ratio):
    # mo
    interval = np.linalg.norm(original_data[1:,0:2]-original_data[0:-1,0:2],axis=1)
    interval[0]=2*interval[0]
    interval = np.cumsum(interval)
    interval = interval/interval[-1]

    displacement = (original_data[-1,:]-original_data[0,:])*distance_ratio
    data = np.zeros_like(original_data)
    data[1:,0]=original_data[0,0]+displacement[0]*interval
    data[1:,1]=original_data[0,1]+displacement[1]*interval
    data[0,:] = original_data[0,:]
    return data

def modeling1(obstacles,original_data,distance_ratio):
    # mo
    n = len(original_data)
    rl=0.1
    ru = 1
    r_array = np.linalg.norm(original_data[1:,0:2]-original_data[0:-1,0:2])
    d = np.sum(r_array)/n
        
    centers,radius = BubbleGeneration(original_data,obstacles,rl,ru)
    opti = ca.Opti()
    Q = opti.variable(n,2)
    D = Q[1:,:]-Q[0:-1,:]
    DD = D[1:,:]-D[0:-1,:]
    opti.minimize(ca.dot(DD,DD))
    sst_dm=ca.DM(original_data)
    opti.subject_to(Q[0,:]==sst_dm[0,0:2])
    # opti.subject_to(Q[1,:]==Q[0,:]+d*ca.DM([[ca.cos(original_data[0,2]),ca.sin(original_data[0,2])]]))
    opti.subject_to(Q[-1,:]==sst_dm[-1,0:2])
    # opti.subject_to(Q[1,:]==sst[0,0:2]+d*ca.tan(sst[0,2]))
    diff = (Q-centers)*(Q-centers)
    opti.subject_to(ca.sum2(diff)<=radius*radius)
    a = ca.DM([3])
    opti.solver('ipopt')
    sol = opti.solve()
    val = np.zeros((n,3))
    val[:,0:2] =  np.array(sol.value(Q))
    return val
    
def BubbleGeneration(sst,obstacles,rl,ru):
    ces_obstacle = Obstacle(obstacles)
    # ces_obstacle.plot_obstacles()
    nearest_pts = ces_obstacle.get_neareset_points(sst)
    # d = np.linalg.norm(sst[1:,0:2]-sst[0:-1,0:2])  
    r_array = np.linalg.norm(sst[:,0:2]-nearest_pts,axis=1)    
    # bubbles=[[sst[0,0],sst[0,1],r_array[0]]]
    centers = [sst[0,0:2]]
    radius = [r_array[0]]
    b = sst[0,0:2]
    r = np.fmin(r_array[0],ru)
    
    for i in range(1,len(nearest_pts)-1):
        if np.linalg.norm(sst[i,0:2]-b)<0.5*r:
            centers.append([b[0],b[1]])
            radius.append(r)
            # bubbles.append([b[0],b[1],r])
            continue
        
        b = sst[i,0:2]
        r = r_array[i]
        if r<rl:
            b = sst[i,0:2] + (rl-r_array[i])*(sst[i,0:2]-nearest_pts[i])/r_array[i]
            r = rl
            new_nearest = ces_obstacle([b])
            new_nearest = np.reshape(new_nearest,(-1,))
            if new_nearest != nearest_pts[i]:
                b = (new_nearest+nearest_pts[i])/2
                r = np.linalg.norm(b-new_nearest)
        centers.append([b[0],b[1]])
        radius.append(r)
        # bubbles.append([b[0],b[1],r])
    
    b = sst[-1,0:2]    
    r = np.fmin(r_array[-1],ru)
    centers.append([b[0],b[1]])
    radius.append(r)
       
    return np.array(centers),np.array(radius) 


def process(sst_data,obstacles,E,A,Iz,rho,distance_ratio=0.9,total_iters=500,dt_per_it=0.2,dt=0.0005,obstacle_force_ratio=500):
    
    sst_data = np.array(sst_data,dtype=np.float32) 
    
    obstacle_force = ObstacleForce(obstacles)    
    node_count = len(sst_data)
    element_count = node_count-1
    
    constraint_index = [0,1,2,3*element_count,3*element_count+1]
    n_dof = 3*node_count-5
    total_dof = 3*node_count


    
    
    # sst_distance = np.linalg.norm(sst_data[:,0:2]-sst_data[0,0:2],axis=1)*distance_ratio
    # positions = np.zeros((node_count,3),dtype=np.float32)
    # positions[:,0] = sst_distance*np.cos(phi0)
    # positions[:,1] = sst_distance*np.sin(phi0)        
    # positions = positions + sst_data[0,:]
    # positions[0,2] = phi0
    
    
    idx1 = np.zeros(3*node_count,dtype=int)
    idx1[0:node_count]=np.arange(0,3*node_count,3)
    idx1[node_count:2*node_count]=np.arange(1,3*node_count+1,3)
    idx1[2*node_count:3*node_count]=np.arange(2,3*node_count+1,3)
    
    M = np.zeros((total_dof,total_dof),dtype=np.float32)
    K = np.zeros((total_dof,total_dof),dtype=np.float32)
    
    K1 = np.zeros((total_dof,total_dof),dtype=np.float32)
    K2 = np.zeros((total_dof,total_dof),dtype=np.float32)
    # yp = np.zeros((2*node_count+1,),dtype=np.float32)
    
    # yp_cp = cp.zeros((2*node_count+1,),dtype=cp.float32)
    
    positions = modeling1(obstacles,sst_data,distance_ratio)
    
    theta0 = sst_data[0,2]-np.arctan2(positions[1,1]-positions[0,1],positions[1,0]-positions[0,0])
    if theta0>np.pi:
        theta0-=2*np.pi
    
    node_list = [Node(i,pos[0:2]) for (i,pos) in enumerate(positions)]
    element_list = [Element(i,node_list[i],node_list[i+1],E[i],A,Iz,rho) for i in range(element_count)]
    
    
    for i,element in enumerate(element_list):
        i1 = 3*node_list.index(element.node1)
        i2 = 3*node_list.index(element.node2)
        k_e = element.global_stiffness_matrix()
        m_e = element.global_mass_matrix()
        
        K[i1:i1+3,i1:i1+3] += k_e[0:3,0:3]
        K[i1:i1+3,i2:i2+3] += k_e[0:3,3:6]
        K[i2:i2+3,i1:i1+3] += k_e[3:6,0:3]
        K[i2:i2+3,i2:i2+3] += k_e[3:6,3:6]

        M[i1:i1+3,i1:i1+3] += m_e[0:3,0:3]
        M[i1:i1+3,i2:i2+3] += m_e[0:3,3:6]
        M[i2:i2+3,i1:i1+3] += m_e[3:6,0:3]
        M[i2:i2+3,i2:i2+3] += m_e[3:6,3:6]
    
    
    K1[0:node_count,:]=K[idx1[0:node_count],:]
    K1[node_count:2*node_count,:]=K[idx1[node_count:2*node_count],:]
    K1[2*node_count:,:]=K[idx1[2*node_count:],:]
    
    
    K2[:,0:node_count]=K1[:,idx1[0:node_count]]
    K2[:,node_count:2*node_count]=K1[:,idx1[node_count:2*node_count]]
    K2[:,2*node_count:]=K1[:,idx1[2*node_count:]]
    
    K2_cp = cp.array(K2,dtype=cp.float32)
    
    
    displacement_history =[]

    

    constraint_deform = np.zeros((3*node_count,),dtype=np.float32)
    constraint_deform[2]=theta0
    constraint_deform[0:2] = sst_data[0,0:2]-positions[0,0:2]
    constraint_deform[-3:-1] = sst_data[-1,0:2]-positions[-1,0:2]

    # print(constraint_deform[-6:])
    
    F_d = np.matmul(K,constraint_deform)
    
    M_del = np.delete(M,constraint_index,axis=0)
    M_del = np.delete(M_del,constraint_index,axis=1)
    K_del = np.delete(K,constraint_index,axis=0)
    K_del = np.delete(K_del,constraint_index,axis=1)
    F_d_del = np.delete(F_d,constraint_index)
    
    # print(M_del)
        
    M_inv = np.linalg.inv(M_del)
    M_inv_root = fractional_matrix_power(M_del,-0.5)  
    C = np.matmul(np.matmul(M_inv_root,K_del),M_inv_root)          
    C = 2*fractional_matrix_power(C,0.5) 
    C.real[abs(C.real)<1e-5]=0.0
    C = np.matrix(C.real,dtype=np.float32)

    # M_cp = cp.array(M_del)
    K_cp = cp.array(K_del)
    M_inv_cp = cp.array(M_inv)
    C_cp = cp.array(C)
    F_d_cp = cp.array(F_d_del,dtype=cp.float32)
    
    
    
    # print(p)
    
      
    its = int(dt_per_it/dt)
    
    integrator_time_elasped = 0.0
    obs_time_elasped = 0.0
    
    positions_cp = cp.array(positions)
    sst_data_cp = cp.array(sst_data)
    
    displacement = cp.zeros((node_count,3),dtype=cp.float32)
    displacement[0,2] = theta0
    displacement[:,0:2] = sst_data_cp[:,0:2]-positions_cp[:,0:2]
    
    y0 =cp.reshape(displacement,(-1,))
    
    y1_cp = cp.zeros((2*n_dof,))
    y1_cp[0:n_dof-1] = y0[3:3*(node_count-1)]
    y1_cp[n_dof-1] = y0[3*(node_count-1)+2]
        
    
    for i in range(total_iters):
        print('{}/{}'.format(i,total_iters))
        
        start = time.time()
        
        obs_force = obstacle_force.get_force(cp.asnumpy(sst_data_cp),k=obstacle_force_ratio)
        obs_force = np.delete(obs_force,constraint_index)
        obs_force_cp = cp.array(obs_force,dtype=cp.float32).reshape((-1,))        
        end = time.time()
        
        obs_time_elasped+=(end-start)
        F = obs_force_cp-F_d_cp
        
        def model(y):
            y_dot = cp.zeros_like(y)
            y_dot[0:n_dof]=y[n_dof:]
            temp = F-cp.matmul(C_cp,y[n_dof:])-cp.matmul(K_cp,y[0:n_dof])
            y_dot[n_dof:] = cp.matmul(M_inv_cp,temp)
            return y_dot 
        
        
              
        
        
        # yp_cp[0:node_count] = displacement[:,0]
        # yp_cp[node_count:2*node_count] = displacement[:,1]
        # yp_cp[2*node_count] = displacement[0,2]
        
        # fp = cp.matmul(K2_cp[2*node_count+1:,0:2*node_count+1],yp_cp)
        # theta = cp.linalg.solve(K2_cp[2*node_count+1:,2*node_count+1:],fp)
        # displacement[1:,2]=theta    
        
       
        y1_cp[n_dof:]=0.0
        start = time.time()
        for j in range(its):
            k1 = model(y1_cp) 
            k2 = model(y1_cp+dt/2*k1)
            k3 = model(y1_cp+dt/2*k2)
            k4 = model(y1_cp+dt*k3)
            y1_cp = y1_cp + dt/6*(k1+2*k2+2*k3+k4)
        end = time.time()
        integrator_time_elasped+=(end-start)
        # print(y1_cp)        
        # yt = cp.asnumpy(y1_cp[0:n_dof])

        # for c in constraint_index:
        #     yt = np.insert(yt,[c],y0[c])
        
        y0[3:3*(node_count-1)]=y1_cp[0:n_dof-1]
        y0[3*(node_count-1)+2]=y1_cp[n_dof-1]
        
        yt = cp.reshape(y0,(-1,3))
        displacement_history.append(cp.asnumpy(yt)) 
        
        sst_data_cp = positions_cp+yt

    print('integration time: ',integrator_time_elasped)
    print('obstacle force time: ',obs_time_elasped)
    return positions,displacement_history

            
if __name__ == "__main__":
    
    script_dir = os.path.dirname(__file__)
    
    # read obstacle file
    obstacles=[]
    obstacle_file = os.path.join(script_dir, 'data/fem_refine/extended_polygon.shp')
    with fiona.open(obstacle_file) as shapefile:
        for record in shapefile:
            geometry = shape(record['geometry'])
            x5,y5 = geometry.exterior.xy
            obstacles.append(np.vstack([x5,y5]).transpose())
    
    # read sst file    
    sst_file = os.path.join(script_dir, 'data/fem_refine/sst_data.txt')
    sst_state = np.array(ca.DM.from_file(sst_file))
    if len(sst_state)%2 == 1:
        sst_state=sst_state[0::2,:]
    else:
        sst_state1 = np.zeros((len(sst_state)/2+1,len(sst_state[0,:])))
        sst_state1[0:-1,:] = sst_state[0::2,:]
        sst_state1[-1,:] = sst_state[-1,:]
        sst_state = sst_state1
    
    
    
    
    r=0.3
    A=2*np.pi*r*r
    Iz =10*np.pi*r*r*r*r
    E=1e7*np.linspace(1,1,len(sst_state)-1)
    # E[0:10] = E[0]
    rho=20000
    total_iters=40
    dt_per_it=0.2
    dt=0.0001
    
    distance_ratio = 1
    obstacle_force_ratio = 2e5
    phi0 = sst_state[0][2]
    
    time_start = time.time()
    positions,displacement_history = process(sst_state,obstacles,E,A,Iz,rho,distance_ratio=distance_ratio,total_iters=total_iters,dt_per_it=dt_per_it,dt=dt,obstacle_force_ratio=obstacle_force_ratio)
    time_end = time.time()  
    print('process time:',time_end-time_start)  
    
    # positions = np.zeros((node_count,3),dtype=np.float32)
    # positions[:,0] = sst_distance*np.cos(phi0)
    # positions[:,1] = sst_distance*np.sin(phi0)        
    # positions = positions + sst_state[0,:]
    # positions[0,2] = phi0
    theta0 = sst_state[0,2]-np.arctan2(sst_state[-1,1]-sst_state[0,1],sst_state[-1,0]-sst_state[0,0])
    if theta0>np.pi:
        theta0-=2*np.pi
    
    ax = plt.subplot(1,1,1)
    obstacle_force = ObstacleForce(obstacles)
    obstacle_force.plot_obstacles(ax)
    # ax.plot(sst_state[0:3,0],sst_state[0:3,1],label='{}'.format(0))
    ax.plot(sst_state[:,0],sst_state[:,1],label='{}'.format(-1))
    ax.plot(positions[:,0],positions[:,1],label='{}'.format(0))
    for i,d in enumerate(displacement_history):
        if (i+1)%5==0:            
            ax.plot(d[:,0]+positions[:,0],d[:,1]+positions[:,1],label='{}'.format(i+1))
            # ax.plot(d[0:3,0]+positions[0:3,0],d[0:3,1]+positions[0:3,1],label='{}'.format(i+1))
            # ax.arrow(d[1,0]+positions[1,0],d[1,1]+positions[1,1],np.cos(d[1,2]-theta0),np.sin(d[1,2]-theta0))
            # ax.arrow(d[2,0]+positions[2,0],d[2,1]+positions[2,1],np.cos(d[2,2]-theta0),np.sin(d[2,2]-theta0))
            print('theta',d[:,2])
    ax.arrow(0,0,np.cos(sst_state[0,2]),np.sin(sst_state[0,2]),width=0.1,label='initial direction')

    plt.legend()
    plt.show()
    
    
    