import enum
from Boundary import Boundary
from Force import Force
from Node import Node
from Element import Element
import numpy as np
from scipy.linalg import fractional_matrix_power
import scipy.linalg
import collections
import casadi as ca

def mesh(nodes_pos:np.array,E,A,I,rho,element_idx=None):

    nodes_count = len(nodes_pos)
    if element_idx is None or len(element_idx)==0:
        element_idx=[[i,i,i+1] for i in range(0,nodes_count-1)]

    node_list = [Node(i,nodes_pos[i]) for i in range(nodes_count)]

    if isinstance(E,(list,np.ndarray)):
        e_array=E
    else:
        e_array=[E]*(nodes_count-1)

    if isinstance(A,(list,np.ndarray)):
        a_array=A
    else:
        a_array=[A]*(nodes_count-1)

    if isinstance(I,(list,np.ndarray)):
        i_array=I
    else:
        i_array=[I]*(nodes_count-1)

    if isinstance(rho,(list,np.ndarray)):
        rho_array=rho
    else:
        rho_array=[rho]*(nodes_count-1)

    element_list=[Element(idx[0],node_list[idx[1]],node_list[idx[2]],e_array[idx[0]],a_array[idx[0]],i_array[idx[0]],rho_array[idx[0]]) for idx in element_idx]

    node_set=set()
    for e in element_list:
        node_set.add(e.node1)
        node_set.add(e.node2) 
    
    sorted_node_list =  sorted(list(node_set), key = lambda e:e.id)

    return sorted_node_list,element_list

def apply_force(node_list,force_list):
    for f in force_list:
        assert(len(f)>1)
        nid = f[0]

        node = filter(lambda node:node.id==nid,node_list)
        node = list(node)
        if not node:
            continue
        node=node[0]

        if isinstance(f[1],Force):
            node.forces.append(f[1])
        else:
            node.forces.append(Force(f[1],f[2],f[3]))

def apply_boundary(node_list,boundary_list):
    for b in boundary_list:
        assert(len(b)>1)
        nid = b[0]
        node = filter(lambda node:node.id==nid,node_list)
        node=list(node)
        if not node:
            continue
        node=node[0]
        if isinstance(b[1],Boundary):
            node.boundaries.append(b[1])
        else:
            node.boundaries.append(Boundary(bool(b[1]),bool(b[2]),bool(b[3])))


def assemble_matrix(sorted_node_list,elements,delete_constraint_columns=True):   
    
    ndof = 3*len(sorted_node_list)

    K = ca.DM.zeros(ndof,ndof)
    M = ca.DM.zeros(ndof,ndof)

    for element in elements:
        i1 = 3*sorted_node_list.index(element.node1)
        i2 = 3*sorted_node_list.index(element.node2)
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

    constraint_index=[]
    F = ca.DM.zeros(ndof)
    for i,node in enumerate(sorted_node_list):
        
        force = Force()
        for f in node.forces:
            force+=f
        
        F[i*3+0]=force.fx
        F[i*3+1]=force.fy
        F[i*3+2]=force.mz

        boundry = Boundary()
        for b in node.boundaries:
            boundry+=b

        if boundry.dx:
            constraint_index.append(i*3)
        if boundry.dy:
            constraint_index.append(i*3+1)
        if boundry.rz:
            constraint_index.append(i*3+2)

    if delete_constraint_columns:
        K.remove(constraint_index,constraint_index)
        M.remove(constraint_index,constraint_index)
        F.remove(constraint_index,[])


    M_inv_root = fractional_matrix_power(np.matrix(M),-0.5)        
    C = 2*fractional_matrix_power(M_inv_root*np.matrix(K)*M_inv_root,0.5)
    M_inv = ca.DM(scipy.linalg.inv(M))

    C.real[abs(C.real)<1e-5]=0.0
    C = ca.DM(C.real)
    

    return M,C,K,F,M_inv,constraint_index

def assemble_matrix_np(sorted_node_list,elements,pre_deform=None,delete_constraint_columns=True):   
    
    ndof = 3*len(sorted_node_list)
    y0 = None
    K = np.zeros([ndof,ndof],dtype=np.float32)
    M = np.zeros([ndof,ndof],dtype=np.float32)

    for element in elements:
        i1 = 3*sorted_node_list.index(element.node1)
        i2 = 3*sorted_node_list.index(element.node2)
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

    constraint_index=[]
    F = np.zeros((ndof,),dtype=np.float32)
    y0 = np.zeros((ndof,),dtype=np.float32)
    for i,node in enumerate(sorted_node_list):
        
        force = Force()
        for f in node.forces:
            force+=f
        
        F[i*3+0]=force.fx
        F[i*3+1]=force.fy
        F[i*3+2]=force.mz

        boundry = Boundary()
        for b in node.boundaries:
            boundry+=b

        if boundry.dx:
            constraint_index.append(i*3)
        if boundry.dy:
            constraint_index.append(i*3+1)
        if boundry.rz:
            constraint_index.append(i*3+2)

    if pre_deform is not None:
        all_deform = np.reshape(pre_deform,(-1,))
        constraint_deform = np.zeros((ndof,),dtype=np.float32)
        constraint_deform[constraint_index]=all_deform[constraint_index]
        F_d = np.matmul(K,constraint_deform)
        F =F-F_d
        
        dm_K = ca.DM(K)
        opti = ca.Opti()
        y_mx = opti.variable(ndof)
        f_mx = opti.variable(ndof)        
        
        opti.subject_to(ca.mtimes(dm_K,y_mx)==f_mx)
        opti.subject_to(y_mx[0::3]==all_deform[0::3])
        opti.subject_to(y_mx[1::3]==all_deform[1::3])
        opti.subject_to(y_mx[2]==all_deform[2])
        opti.subject_to(f_mx[5::3]==0)
        opti.solver('ipopt')
        opti.solve()
        y0 = opti.value(y_mx)
        
        print('y0 = ',y0)   
        print('f0 = ',opti.value(f_mx))  
        print('F = ',F) 
        y0 = np.array(y0,dtype=np.float32)
        
    if delete_constraint_columns:
        K = np.delete(K,constraint_index,0)
        K = np.delete(K,constraint_index,1)
        # K = np.matrix(K)
        
        M = np.delete(M,constraint_index,0)
        M = np.delete(M,constraint_index,1)
        # M = np.matrix(M)
        
        F = np.delete(F,constraint_index,0)
        # F = np.matrix(F)
        
        
        y0 = np.delete(y0,constraint_index,0)
        # y0 = np.matrix(y0)
        
    M_inv_root = fractional_matrix_power(M,-0.5)        
    C = 2*fractional_matrix_power(M_inv_root*K*M_inv_root,0.5)
 
    C.real[abs(C.real)<1e-5]=0.0
    C = np.matrix(C.real,dtype=np.float32)
    

    return M,C,K,F,y0,constraint_index

if __name__ == '__main__':
    n1 = Node(0,np.array([1,1.0]))
    n2 = Node(1,np.array([-1,3.0]))
    s=set()
    s.add(n1)
    s.add(n2)
    n3 = n1
    s.add(n3)