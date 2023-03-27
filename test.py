from Node import Node
from Element import Element
from beam_fem import *
import casadi as ca
import matplotlib.pyplot as plt
from fem_refine import *

import fiona
from shapely.geometry import shape

def load_obastacle():

    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }
    obstacles=[]
    with fiona.open("extended_polygon.shp") as shapefile:
        for record in shapefile:
            geometry = shape(record['geometry'])
            x5,y5 = geometry.exterior.xy
            obstacles.append(np.vstack([x5,y5]).transpose())
    return obstacles
    #         plt.plot(x5,y5)
    # plt.show()

def test_2_beams():
    node_pos = np.array([[0,0],[0,0.5],[0,1.0]])
    r=0.3
    sorted_node_list,element_list = mesh(node_pos,1e6,A=np.pi*r*r,I=np.pi*r**4,rho=200)

    for e in element_list:
        print('element:',e.id,';node1:',e.node1.id,';node2:',e.node2.id,'length:',e.length)

    apply_force(sorted_node_list,[[2,0,2000,0]])

    apply_boundary(sorted_node_list,[[0,True,True,True]])
    
    M,C,K,F,constraint_index = assemble_matrix(sorted_node_list,element_list,delete_constraint_columns=False)

    print('constraint_index',constraint_index)  
    M_inv = ca.inv(M)
    print('M=',M)
    print('K=',K)
    print('M_inv=',M_inv)
    print('F=',F)

    ndof = M.columns()
    y1 = ca.MX.sym("y1",ndof)
    y2 = ca.MX.sym("y2",ndof)
    Z = ca.MX.sym("z",9)
    
    # force =ca.DM.zeros(ndof-3)
    # force[4]=2000
    # F=ca.vertcat(Z,force)
    # force = ca.vertcat(Z,F[3:])

    y1_dot = y2
    rhs=Z-ca.mtimes(K,y1)
    # rhs=F-ca.mtimes(C,y2)-ca.mtimes(K,y1)
    y2_dot = ca.mtimes(M_inv,rhs)
    x =ca.vertcat(y1,y2)
    # z = F[0:3]
    ode=ca.vertcat(y1_dot,y2_dot)

    dae = {'x':x, 'z':Z, 'ode':ode, 'alg':ca.vertcat(y1[0:3],Z[3:]-F[3:])}
    option={}
    option['tf']=0.5
    

    I = ca.integrator('I', 'idas', dae,option)

    #y1_0=ca.DM([0,0,0,0,0.5,0,0,1.0,0])
    y1_0=ca.DM.zeros(9,1)
    # y1_0[4]=0.1
    # y1_0[5]=-1
    # y1_0[5]=-1
    y2_0=ca.DM.zeros(9,1)
    I(x0=ca.vertcat(y1_0,y2_0))

def test_2_beams_no_constraint():
    node_pos = np.array([[0,0],[0,0.5],[0,1.0]])
    r=0.3
    sorted_node_list,element_list = mesh(node_pos,1e6,A=np.pi*r*r,I=np.pi*r**4,rho=200)

    for e in element_list:
        print('element:',e.id,';node1:',e.node1.id,';node2:',e.node2.id,'length:',e.length)

    
    apply_force(sorted_node_list,[[2,0,2000,0]])

    apply_boundary(sorted_node_list,[[0,True,True,True]])
    
    M,C,K,F,constraint_index = assemble_matrix(sorted_node_list,element_list,delete_constraint_columns=True)

    print('constraint_index',constraint_index)  


    M_inv = ca.inv(M)
    print('M=',M)
    print('K=',K)
    print('M_inv=',M_inv)
    print('F=',F)

    ndof = M.columns()
    y1 = ca.MX.sym("y1",ndof)
    y2 = ca.MX.sym("y2",ndof)
    # Z = ca.MX.sym("z",3)
    
    F =ca.DM.zeros(ndof)
    F[0]=2000
    # F=ca.vertcat(Z,force)

    y1_dot = y2
    rhs=F-ca.mtimes(K,y1)
    # rhs=F-ca.mtimes(C,y2)-ca.mtimes(K,y1)
    y2_dot = ca.mtimes(M_inv,rhs)
    x =ca.vertcat(y1,y2)
    # z = F[0:3]
    ode=ca.vertcat(y1_dot,y2_dot)

    dae = {'x':x, 'ode':ode}
    option={}
    option['tf']=0.1
    

    I = ca.integrator('I', 'idas', dae,option)

    #y1_0=ca.DM([0,0,0,0,0.5,0,0,1.0,0])
    y1_0=ca.DM.zeros(6,1)
    # y1_0[4]=0.1
    # y1_0[5]=-1
    # y1_0[5]=-1
    y2_0=ca.DM.zeros(6,1)
    r = I(x0=ca.vertcat(y1_0,y2_0))
    xt = r['xf']
    print(xt[0:6])

def test_vibration():

    M = ca.diag([1,1,1,1])
    print('M:',M)

    k1=k2=k3=10
    K = ca.diag([k1,k1+k2,k2+k3,k3])
    K[1,0]=-k1
    K[0,1]=-k1

    K[1,2]=-k2
    K[2,1]=-k2

    K[2,3]=-k3
    K[3,2]=-k3

    print('K:',K)
    # node_pos = np.array([[0,0],[0,0.5],[0,1.0]])
    # r=1000
    # node_list,element_list = mesh(node_pos,1e6,A=np.pi*r*r,I=np.pi*r**4,rho=2000)

    # for e in element_list:
    #     print('element:',e.id,';node1:',e.node1.id,';node2:',e.node2.id,'length:',e.length)

    # M,C,K = assemble_matrix(element_list)

    # M_inv = np.linalg.inv(M)
    # # print("M = \n{}".format(M))

    # C.real[abs(C.real)<1e-5]=0.0

    # # print(np.real(C))
    # # print(K)
    # M = ca.DM(M)
    # M_inv = ca.DM(M_inv)
    # C = ca.DM(C.real)
    # K = ca.DM(K) 
    M_inv = ca.inv(M)

    

    ndof = M.columns()
    y1 = ca.MX.sym("y1",ndof)
    y2 = ca.MX.sym("y2",ndof)
    Z = ca.MX.sym("z",2)

    force = ca.DM.zeros(ndof-2)
    force[0] = 3
    F=ca.vertcat(Z[0],force,Z[1])

    y1_dot = y2
    rhs=F-ca.mtimes(K,y1)
    # rhs=F-ca.mtimes(C,y2)-ca.mtimes(K,y1)
    y2_dot = ca.mtimes(M_inv,rhs)
    x =ca.vertcat(y1,y2)
    # z = F[0:3]
    ode=ca.vertcat(y1_dot,y2_dot)

    dae = {'x':x, 'z':Z, 'ode':ode, 'alg':ca.vertcat(y1[0],y1[-1])}

    

    #y1_0=ca.DM([0,0,0,0,0.5,0,0,1.0,0])
    y1_0=ca.DM.zeros(4,1)
    # y1_0[2]=0.2
    y2_0=ca.DM.zeros(4,1)

    tsim = np.linspace(0,10.0,1000)
    integrator_options={}
    integrator_options['grid'] = tsim
    integrator_options['output_t0'] = True

    I = ca.integrator('integrator', 'idas', dae, {'grid':tsim, 'output_t0':True})

    r = I(x0=ca.vertcat(y1_0,y2_0))
    xt = r['xf']
    print(xt.size())
    plt.figure(1)
    plt.plot(tsim,xt[0,:].T,label="x1")
    plt.plot(tsim,xt[1,:].T,label="x2")
    plt.plot(tsim,xt[2,:].T,label="x3")
    plt.plot(tsim,xt[3,:].T,label="x4")
    plt.title('position')
    plt.legend()

    plt.figure(2)
    plt.plot(tsim,xt[4,:].T,label="x1_dot")
    plt.plot(tsim,xt[5,:].T,label="x2_dot")
    plt.plot(tsim,xt[6,:].T,label="x3_dot")
    plt.plot(tsim,xt[7,:].T,label="x4_dot")
    plt.title('velocity')
    plt.legend()
    plt.show()

    # print(r)
def test_fem_refine():
    obs = load_obastacle()
    fem = FemRefine(obs)
    fem.obstacls_force.plot_obstacles()


if __name__=='__main__':
    test_fem_refine()
    # test_2_beams_no_constraint()
    # test_vibration()
    # test_2_beams()