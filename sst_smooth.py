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

def modeling(original_data,distance_ratio):
    interval = original_data[1:,0:2]-original_data[0:-1,0:2]
    d_original = np.linalg.norm(interval,axis=1)
    phi_original = np.arctan2(interval[:,1],interval[:,0])
    dphi = np.zeros_like(phi_original)
    dphi[1:] = (phi_original[1:]-phi_original[0:-1])/2.0
    
    dphi[0] = original_data[0,2]
    
    phi = np.cumsum(dphi)
    data = np.zeros_like(original_data)
    data[0,:] = original_data[0,:]
    
    # displacement = np.zeros((len(d_original),2))
    data[1:,0] = d_original*np.cos(phi)*distance_ratio
    data[1:,1] = d_original*np.sin(phi)*distance_ratio
    data = np.cumsum(data,axis=0)
    data[1:,2]=phi[:]
    return data

def modeling2(original_data,distance_ratio):
    interval = original_data[1:,0:2]-original_data[0:-1,0:2]
    d_original = np.linalg.norm(interval,axis=1)
    phi_original = np.arctan2(interval[:,1],interval[:,0])
    dphi = np.zeros_like(phi_original)
    dphi[1:] = (phi_original[1:]-phi_original[0:-1])/2.0
    
    dphi[0] = original_data[0,2]
    
    phi = np.cumsum(dphi)
    data = np.zeros_like(original_data)
    data[0,:] = original_data[0,:]
    
    # displacement = np.zeros((len(d_original),2))
    data[1:,0] = d_original*np.cos(phi)*distance_ratio
    data[1:,1] = d_original*np.sin(phi)*distance_ratio
    data = np.cumsum(data,axis=0)
    data[1:,2]=phi[:]
    return data

# def modeling1(original_data,distance_ratio):
#     interval = original_data[1:,0:2]-original_data[0:-1,0:2]
#     d_original = np.linalg.norm(interval,axis=1)*distance_ratio
#     data = np.zeros_like(original_data)
#     data[0,:] = original_data[0,:]
#     for i in range(1,len(data)):
#         theta = data[i-1,2]
#         c = np.cos(theta)
#         s = np.sin(theta)
#         tm = np.array([[c,s],[-s,c]])
#         p = data[i,0:2]-data[i-1,0:2]
#         p = np.matmul(tm,p)
#         theta = np.arctan2(p[1],p[0])
#         data[i,0]=data[i-1,0]+d_original[i-1]*c
#         data[i,1]=data[i-1,1]+d_original[i-1]*s
#         data[i,2]=data[i-1,2]+theta/2
    
#     return data


def process(sst_data,obstacles,E,A,Iz,rho,distance_ratio=0.5,total_iters=500,dt_per_it=0.2,dt=0.0005):
    
    sst_data = np.array(sst_data,dtype=np.float32) 
    
    obstacle_force = ObstacleForce(obstacles)    
    node_count = len(sst_data)
    element_count = node_count-1
    
    constraint_index = [0,1,2,3*element_count,3*element_count+1]
    n_dof = 3*node_count-5
    total_dof = 3*node_count
    phi0 = sst_data[0,2]
    
    
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
    
    
    
    displacement_history =[]
    position_history=[]
    for i in range(total_iters):
        print('{}/{}'.format(i,total_iters))
        
        M = np.zeros((total_dof,total_dof),dtype=np.float32)
        K = np.zeros((total_dof,total_dof),dtype=np.float32)
        K1 = np.zeros((total_dof,total_dof),dtype=np.float32)
        K2 = np.zeros((total_dof,total_dof),dtype=np.float32)
        yp = np.zeros((2*node_count+1,),dtype=np.float32)
        
        positions = modeling(sst_data,distance_ratio)
        
        position_history.append(positions)
        
        displacement = np.zeros((node_count,3),dtype=np.float32)
        displacement[:,0:2] = sst_data[:,0:2]-positions[:,0:2]
        displacement[0,2] = positions[0,2]
        
        node_list = [Node(i,pos[0:2]) for (i,pos) in enumerate(positions)]
        element_list = [Element(i,node_list[i],node_list[i+1],E,A,Iz,rho) for i in range(element_count)]

        
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
        
        
        yp[0:node_count] = displacement[:,0]
        yp[node_count:2*node_count] = displacement[:,1]
        yp[2*node_count] = displacement[0,2]
        
        fp = np.matmul(K2[2*node_count+1:,0:2*node_count+1],yp)
        theta = np.linalg.solve(K2[2*node_count+1:,2*node_count+1:],fp)
        
        # displacement[1:,2]=theta
        y0 =np.reshape(displacement,(-1,))    
        
        constraint_deform = np.zeros((3*node_count,))
        constraint_deform[constraint_index]=y0[constraint_index]
        F_d = np.matmul(K,constraint_deform)
        
        M_del = np.delete(M,constraint_index,axis=0)
        M_del = np.delete(M_del,constraint_index,axis=1)
        K_del = np.delete(K,constraint_index,axis=0)
        K_del = np.delete(K_del,constraint_index,axis=1)
        F_d_del = np.delete(F_d,constraint_index)
        
        print(M_del)
            
        M_inv = np.linalg.inv(M_del)
        M_inv_root = fractional_matrix_power(M_del,-0.5)  
        C = np.matmul(np.matmul(M_inv_root,K_del),M_inv_root)          
        C = 2*fractional_matrix_power(C,0.5) 
        C.real[abs(C.real)<1e-5]=0.0
        C = np.matrix(C.real,dtype=np.float32)
    
        M_cp = cp.array(M_del)
        K_cp = cp.array(K_del)
        M_inv_cp = cp.array(M_inv)
        C_cp = cp.array(C)
        F_d_cp = cp.array(F_d_del)
        
        
        v0 = cp.zeros((n_dof,))
    
        
        p = np.reshape(y0,(-1,3))+positions
        # print(p)
        
        # obs_force = obstacle_force.get_force(p,k=2.0)
        # obs_force = np.delete(obs_force,constraint_index)
        
        y1 = np.delete(np.reshape(y0,(-1,3)),constraint_index)
        
        # obs_force_cp = cp.array(obs_force).reshape((-1,))
        F = cp.array(0-F_d_cp,dtype=cp.float32)
        
                
        def model(y):
            y_dot = cp.zeros_like(y)
            y_dot[0:n_dof]=y[n_dof:]
            temp = F-cp.matmul(C_cp,y[n_dof:])-cp.matmul(K_cp,y[0:n_dof])
            y_dot[n_dof:] = cp.matmul(M_inv_cp,temp)
            return y_dot   
        
        y1_cp = cp.append(cp.array(y1),v0)
        
        its = int(dt_per_it/dt)
        for j in range(its):
            k1 = model(y1_cp) 
            k2 = model(y1_cp+dt/2*k1)
            k3 = model(y1_cp+dt/2*k2)
            k4 = model(y1_cp+dt*k3)
            y1_cp = y1_cp + dt/6*(k1+2*k2+2*k3+k4)
        print(y1_cp)
            
        yt = cp.asnumpy(y1_cp[0:n_dof])

        for c in constraint_index:
            yt = np.insert(yt,[c],y0[c])
        # v0 = y1_cp[n_dof:] 
        
        yt = np.reshape(yt,(-1,3))
        displacement_history.append(yt) 
        
        sst_data = positions+yt
        
        # ax = plt.subplot(1,1,1)
        # obstacle_force.plot_obstacles(ax)
        # ax.plot(positions[:,0],positions[:,1])
        # ax.plot(sst_data[:,0],sst_data[:,1])
        # plt.show()

        # data.append(yt)
    return position_history,displacement_history

            
if __name__ == "__main__":
    
    script_dir = os.path.dirname(__file__)
    
    # read obstacle file
    obstacles=[]
    obstacle_file = os.path.join(script_dir, 'extended_polygon.shp')
    with fiona.open(obstacle_file) as shapefile:
        for record in shapefile:
            geometry = shape(record['geometry'])
            x5,y5 = geometry.exterior.xy
            obstacles.append(np.vstack([x5,y5]).transpose())
    
    # read sst file    
    sst_file = os.path.join(script_dir, 'racecar_planner_temp/sst_data.txt')
    sst_state = np.array(ca.DM.from_file(sst_file))
    
    
    
    r=0.3
    A=10*np.pi*r*r
    Iz =np.pi*r*r*r*r
    E=1e4
    rho=20000
    total_iters=50
    dt_per_it=0.1
    dt=0.0001
    
    distance_ratio = 0.8
    phi0 = sst_state[0][2]
    
    position_history,displacement_history = process(sst_state,obstacles,E,A,Iz,rho,distance_ratio=distance_ratio,total_iters=total_iters,dt_per_it=dt_per_it,dt=dt)
        
    
    # positions = np.zeros((node_count,3),dtype=np.float32)
    # positions[:,0] = sst_distance*np.cos(phi0)
    # positions[:,1] = sst_distance*np.sin(phi0)        
    # positions = positions + sst_state[0,:]
    # positions[0,2] = phi0
    
    
    
    ax = plt.subplot(1,1,1)
    obstacle_force = ObstacleForce(obstacles)
    obstacle_force.plot_obstacles(ax)
    
    for i,(p,d) in enumerate(zip(position_history,displacement_history)):
        if (i+1)%10==0:            
            ax.plot(d[:,0]+p[:,0],d[:,1]+p[:,1],label='{}'.format(i+1))

    plt.legend()
    plt.show()
    
    
    