#!/usr/bin/env python

import copy
import os
from tkinter import N
from ObstacleForce import *
import casadi
import numpy as np
from beam_fem import *
import matplotlib.pyplot as plt
import scipy.integrate
import scipy
import fiona
from shapely.geometry import shape

def apply_streching_force(element_list,k=1000.0):
    for element in element_list:
        angle = element.angle
        mag = k/element.length
        element.node1.forces.append(Force(mag*np.cos(angle),mag*np.sin(angle),0))
        element.node2.forces.append(Force(mag*np.cos(angle),-mag*np.sin(angle),0))

class FemRefine:
    def __init__(self,obstacles) -> None:
        self.obstacls_force = ObstacleForce(obstacles)
        self.plot_history = []
        self.element_history = []
         
        
    def process(self,init_state,E,A,I,rho,total_iterations,tf,stretching_coeff=1000.0,obstacle_coeff=2e6):
        state = init_state       
        if len(init_state[0]==2):
            phi0 =np.arctan2(init_state[1,1]-init_state[0,1],init_state[1,0]-init_state[0,0])        
        else:
            phi0 = init_state[0,2] 
                   
        n_elem = len(init_state)-1        
        self.plot_history.append(np.copy(state[:,0:2]))
        
        direction = np.array([np.cos(phi0), np.sin(phi0)])
        direction_dm = casadi.DM(direction)
        
        for iter in range(total_iterations):
            positions = np.zeros((n_elem+1,2))
            positions[2:,:] = state[2:,0:2]
            positions[0,:] = state[0,0:2]            
            
            p0 = casadi.DM(positions[0,:])
            p2 = casadi.DM(positions[2,:])
            
            d1 = casadi.norm_2(p2-p0)

            opti = casadi.Opti()
            p1 = opti.variable(2,1)
            d0 = casadi.norm_2(p2-p1)
            d2 = casadi.norm_2(p0-p1)
            opti.subject_to((p1-p0)/d2==direction_dm)
            opti.subject_to(d2>0)
            opti.minimize(casadi.dot(d0-d1,d0-d1)+casadi.dot(d1-d2,d1-d2)+casadi.dot(d2-d0,d2-d0))
            opti.set_initial(p1,p0+direction_dm*d1/2)
            casadi_opts={'print_time':False}
            opopt_opts = {'print_level':0}
            opti.solver('ipopt',casadi_opts,opopt_opts)
            sol = opti.solve()
            print('Insert the direction guide point: {}'.format(sol.value(p1)))
            positions[1,:] = sol.value(p1)
            
            state,nodes,elements = self.simulate(positions,E,A,I,rho,tf,stretching_coeff,obstacle_coeff);
            self.plot_history.append(np.copy(state))
            self.element_history.append(elements)
        return state
            
    
    def simulate(self,positions,E,A,I,rho,tf,stretching_coeff,obstacle_coeff):
        
        n_elem = len(positions)-1
        sorted_node_list,element_list = mesh(positions,E,A,I,rho)

        for e in element_list:
            print('element:',e.id,';node1:',e.node1.id,';node2:',e.node2.id,'length:',e.length,'direction',e.angle)
            
        apply_streching_force(element_list,stretching_coeff)
        self.obstacls_force.apply_forces(node_list=sorted_node_list,k=obstacle_coeff)
        # constraints
        apply_boundary(sorted_node_list,[[0,True,True,True],[n_elem,True,True,False]])
        
        M,C,K,F,M_inv,constraint_index = assemble_matrix(sorted_node_list,element_list,delete_constraint_columns=True)

        print('constraint_index',constraint_index)  

        # M_inv = ca.inv(M)
        

        ndof = M.columns()
        #y1=x,y2=x_dot
        y1 = ca.MX.sym("y1",ndof)
        y2 = ca.MX.sym("y2",ndof)
        
        #model
        y1_dot = y2
        # rhs=F-ca.mtimes(K,y1)
        rhs=F-ca.mtimes(C,y2)-ca.mtimes(K,y1)
        y2_dot = ca.mtimes(M_inv,rhs)

        x =ca.vertcat(y1,y2)
        ode=ca.vertcat(y1_dot,y2_dot)

        dae = {'x':x, 'ode':ode}
        option={}
        option['tf']=tf       

        Inte = ca.integrator('Inte', 'idas', dae,option)


        y1_0=ca.DM.zeros(ndof,1)
        y2_0=ca.DM.zeros(ndof,1)

        r = Inte(x0=ca.vertcat(y1_0,y2_0))
        xt = r['xf'][0:ndof]
        print('displacement:',xt)
        pts = np.array([n_elem+1,])
        
        j=0
        for i in range(len(pts)):
            if i in constraint_index:
                continue
            pts[i]=xt[j]
            j+=1
            
        dx=np.array(casadi.reshape(pts,(3,)).T()[:,0:2])
        return dx+positions,sorted_node_list,element_list
        
    def plot_history(self,interval=10):
        n = 0
        total = len(self.plot_history)
        while(n<total):
            plt.plot(self.plot_history[n][:,0],self.plot_history[n][:,1])
            
        plt.show()
        
    def plot_force(self,index,ratio=None):
        if ratio is None:
            l = 0
            for e in self.element_history[index]:
                l+=e.length
            ratio = l/len(self.element_history[index])
            
        for e in self.element_history[index]:
            plt.plot([e.node1.pos[0],e.node2.pos[0]],[e.node1.pos[1],e.node2.pos[1]],'-r')
            for f in e.node1.forces:
                mag = np.linalg.norm([f.fx,f.fy])
                plt.arrow(e.node1.pos[0],e.node1.pos[1],f.fx/mag*ratio,f.fy/mag*ratio)
            
      
class FemRefineScipy:
    def __init__(self,init_state,phi0, obstacles_force,E,A,I,rho,stretching_coeff,obstacle_coeff) -> None:
        self.obstacls_force = obstacles_force
        
        self.plot_history = []
        self.element_history = []
        self.positions = init_state[:,0:2]
        n_elem = len(init_state)-1
        self.phi0 = phi0
        
        self.sorted_node_list,self.element_list = mesh(init_state,E,A,I,rho)
        # apply_streching_force(self.element_list,stretching_coeff)
        self.obstacls_force.apply_forces(node_list=self.sorted_node_list,k=obstacle_coeff)
        apply_boundary(self.sorted_node_list,[[0,True,True,True],[n_elem,True,True,False]])
        self.M,self.C,self.K,self.F,self.y0,self.constraint_index = assemble_matrix_np(self.sorted_node_list,self.element_list,delete_constraint_columns=True)
        self.n_dof = 3*(n_elem+1)-len(self.constraint_index)
        self.M_inv = np.matrix(scipy.linalg.inv(self.M))
        
        
        
        print("constructor finished")
        
    def model(self,y,t):
        y_dot = np.zeros_like(y)
        y_dot[0:self.n_dof]=y[self.n_dof:]
        # temp = self.F-np.matmul(self.C,y[self.n_dof:])-np.matmul(self.K,y[0:self.n_dof])
        temp = self.F-np.matmul(self.K,y[0:self.n_dof])
        y_dot[self.n_dof:] = np.matmul(self.M_inv,temp.transpose()).reshape(self.n_dof,)
        return y_dot
    
    def simulator(self):
        # state = np.insert(self.positions,[2],np.zeros((len(self.positions),1)),axis=1)
        # state = state.reshape((3*len(state)))
        # y0 = np.zeros([2*self.n_dof,])
        y0 = np.zeros([self.n_dof+len(self.constraint_index)])
        previous_phi = self.phi0
        
        for i in range(len(self.element_list)-1):
            
            ux = self.element_list[i].length*.1
            phi = self.element_list[i].angle
            shrienk = np.array([ux,0.0])
            mat = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
            y0[(i+1)*3:(i+1)*3+2] = np.matmul(mat,shrienk)
            
            angle = self.element_list[i+1].angle-previous_phi
            previous_phi = self.element_list[i].angle
            if angle<-np.pi:
                angle=angle+2*np.pi
            elif angle>np.pi:
                angle = angle-2*np.pi
                
            y0[(i+1)*3+2]=angle
        
        y0 = np.delete(y0,self.constraint_index)
        y0 = np.append(y0,np.zeros(self.n_dof,))    
        print('y0 = ',y0)    
        
        y_list =scipy.integrate.odeint(self.model,y0,t=np.arange(0,1,0.01))
        
        y_list = y_list[:,0:self.n_dof]
        
        z = np.zeros((y_list.shape[0],1))
        for c in self.constraint_index:
            y_list = np.insert(y_list,[c],z,axis=1)
             
        # y = y_list[0,:]
        # y = np.reshape(y,(-1,3))
        # plt.plot(y[:,0]+self.positions[:,0],y[:,1]+self.positions[:,1],label='{}'.format(0))
        # y = y_list[-1,:]
        # y = np.reshape(y,(-1,3))
        # plt.plot(y[:,0]+self.positions[:,0],y[:,1]+self.positions[:,1],label='{}'.format(1))
        # # for i,y in enumerate(y_list):
        # #     # if i%3==0:
        # #     y = np.reshape(y,(-1,3))
        # #     plt.plot(y[:,0]+self.positions[:,0],y[:,1]+self.positions[:,1],label='{}'.format(i))
        # # # print(type(y_list))
        # # # print(y_list.shape)
        # plt.legend()
        # plt.show()
        return y_list   
        


def process(init_state,obstacles,E,A,I,rho,stretching_coeff=1000.0,obstacle_coeff=2e7):
    """_summary_

    Args:
        init_state (list): the initial states [x,y,theta] from sst. Note only the theta of the first state is used.
        obstacls (list of polygon): obstacle list
        plot (bool, optional): can be ignored. Defaults to False.
        

    Returns:
        list of states: the final states
    """

    obstacls_force = ObstacleForce(obstacles,obstacle_coeff)

    state=init_state

    # n_elem = len(state)-1
    # positions = np.zeros((3,n_elem+1))
    # positions[0:2,:] = state.T[0:2,:]

    n_elem = len(state)-1
    positions = np.zeros((n_elem+1,2))
    positions[2:,:] = state[2:,0:2]
    positions[0,:] = state[0,0:2]
    direction = np.array([np.cos(phi0), np.sin(phi0)])
    direction_dm = casadi.DM(direction)


    total_its = 40
    plot_lines = 4
    tf = 0.1
    save_indice = int(total_its/plot_lines)
    total_steps = 400
    total_time = 4.0

    if len(state[0]==2):
        phi0 =np.arctan2(state[1,1]-state[0,1],state[1,0]-state[0,0])
        positions[1,:]=state[1,0:2]
        
    else:
        phi0 = state[0,2]

    for i in range(0,total_its+1):
        print('iter = {}/{}'.format(i,total_its))



        p0 = casadi.DM(positions[0,:])
        p2 = casadi.DM(positions[2,:])
        
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
        plot_history=[np.copy(positions)]

        # print(sol.value(casadi.dot(d0-d1,d0-d1)+casadi.dot(d1-d2,d1-d2)+casadi.dot(d2-d0,d2-d0)))
        # print(positions[:,0:3])
        sorted_node_list,element_list = mesh(positions,E,A,I,rho)
        sorted_node_list=[]
        for e in element_list:
            print('element:',e.id,';node1:',e.node1.id,';node2:',e.node2.id,'length:',e.length,'direction',e.angle)
        #     sorted_node_list.append(e.node1)
        
        # sorted_node_list.append(e.node2)
            
        apply_streching_force(element_list,stretching_coeff)
        obstacls_force.apply_forces(node_list=sorted_node_list)
        # constraints
        apply_boundary(sorted_node_list,[[0,True,True,True],[n_elem,True,True,False]])
        
        M,C,K,F,constraint_index = assemble_matrix(sorted_node_list,element_list,delete_constraint_columns=True)

        print('constraint_index',constraint_index)  

        M_inv = ca.inv(M)
        print('M=',M)
        print('K=',K)
        print('M_inv=',M_inv)
        print('F=',F)

        ndof = M.columns()
        #y1=x,y2=x_dot
        y1 = ca.MX.sym("y1",ndof)
        y2 = ca.MX.sym("y2",ndof)
        
        #model
        y1_dot = y2
        rhs=F-ca.mtimes(K,y1)
        y2_dot = ca.mtimes(M_inv,rhs)

        x =ca.vertcat(y1,y2)
        ode=ca.vertcat(y1_dot,y2_dot)

        dae = {'x':x, 'ode':ode}
        option={}
        option['tf']=tf       

        I = ca.integrator('I', 'idas', dae,option)


        y1_0=ca.DM.zeros(ndof,1)
        y2_0=ca.DM.zeros(ndof,1)

        r = I(x0=ca.vertcat(y1_0,y2_0))
        xt = r['xf'][0:ndof]
        print(xt)
        positions=casadi.reshape(xt,(3,)).T()[:,0:2]
        
def test0():
    obstacles=[]
    with fiona.open("extended_polygon.shp") as shapefile:
        for record in shapefile:
            geometry = shape(record['geometry'])
            x5,y5 = geometry.exterior.xy
            obstacles.append(np.vstack([x5,y5]).transpose())

    obstacle_force = ObstacleForce(obstacles)
    
    # read sst
    script_dir = os.path.dirname(__file__)
    sst_file = os.path.join(script_dir, 'racecar_planner_temp/sst_data.txt')
    sst_state = np.array(ca.DM.from_file(sst_file))
    
    r=0.3
    A=np.pi*r*r
    Iz =np.pi*r*r*r*r
    E=1e6
    rho=2000
    total_its = 2
    phi0 = sst_state[0,2] 
    positions = sst_state[:,0:2]
    print(positions[-1])
    plot_history = [copy.copy(positions)]
    
    for iter in range(total_its):
        print('processing: {}/{}'.format(iter,total_its))      
        sim = FemRefineScipy(positions,phi0,obstacle_force,E,A,Iz,rho,stretching_coeff=100.0,obstacle_coeff=2e6)
        y_list = sim.simulator()
        sst_state = np.reshape(y_list[-1,:],(-1,3))
        print(sst_state)
        positions = positions + sst_state[:,0:2]
        plot_history.append(copy.copy(positions))
    
    ax = plt.subplot(1,1,1)
    obstacle_force.plot_obstacles(ax)
    for i,y in enumerate(plot_history):
        # if i%3==0:
            # y = np.reshape(y,(-1,3))
        ax.plot(y[:,0],y[:,1],label='{}'.format(i))
    ax.legend()
    plt.show()
def test1():
    p = np.arange(0,1.0,0.05)
    print(p)
    node_pos = np.zeros((len(p),2))
    node_pos[:,1]=p[:]
    print(node_pos)
    # node_pos = np.array([[0,0],[0,0.1],[0,0.2],[0,0.3],[0,0.4]],dtype=np.float32)
    r=0.2
    sorted_node_list,element_list = mesh(node_pos,E=1e6,A=np.pi*r*r,I=np.pi*r**4,rho=2000)

    for e in element_list:
        print('element:',e.id,';node1:',e.node1.id,';node2:',e.node2.id,'length:',e.length)

    # apply_force(sorted_node_list,[[2,0,2000,0]])

    apply_boundary(sorted_node_list,[[0,True,True,True],[len(node_pos)-1,True,True,False]])

    
    pre_deform=np.zeros((len(node_pos),3))
    pre_deform[0,2]=-np.pi/3
    pre_deform[-1,:]=np.array([0.08,0.2,0])
    # pre_deform=np.array([[0,0,-np.pi/3],[0,0,0],[0,0,0],[0.09,0.1,0],[0.08,0.2,0]],dtype=np.float32)    
    y0 = pre_deform.reshape((-1,))
    
    
    M,C,K,F,y2,constraint_index = assemble_matrix_np(sorted_node_list,element_list,pre_deform,delete_constraint_columns=True)
    M_inv = scipy.linalg.inv(M)

    print('constraint_index',constraint_index)  
    print('F=',F)
    n_dof = 3*len(node_pos)-len(constraint_index)
    

    y1 = np.delete(y0,constraint_index)
    
    print('y1=',y1)
    print('y2=',y2)
    y1 = np.append(y1,np.zeros((n_dof,)))
    
    def model(y,t):
        y_dot = np.zeros_like(y)
        y_dot[0:n_dof]=y[n_dof:]
        temp = F-np.matmul(C,y[n_dof:])-np.matmul(K,y[0:n_dof])
        # temp = F-np.matmul(K,y[0:n_dof])
        y_dot[n_dof:] = np.matmul(M_inv,temp.transpose()).reshape(n_dof,)
        return y_dot   

    # yt = model(y1,0)
    # print('yt:',yt)

    # temp = self.F-np.matmul(self.C,y[self.n_dof:])-np.matmul(self.K,y[0:self.n_dof])

    y_list =scipy.integrate.odeint(model,y1,t=np.arange(0,10,0.0005))
    print('yt=',y_list[-1])
    
    y_list = y_list[:,0:n_dof]    
    z = np.zeros((y_list.shape[0],1))
    for c in constraint_index:
        y_list = np.insert(y_list,[c],y0[c]*np.ones((y_list.shape[0],1)),axis=1)
        
    interval = int(len(y_list)/10)
    for i,y in enumerate(y_list):
        if i%interval==0:
            dy = np.reshape(y,(-1,3))
            pos = node_pos + dy[:,0:2]
            plt.plot(pos[:,0],pos[:,1],label='{}'.format(i))
    
    
        
    print(pos)
    # plt.plot(pos[:,0],pos[:,1],label='{}'.format(-1))
    plt.legend()
    plt.show()     
if __name__ == "__main__":
    test1()

        
    