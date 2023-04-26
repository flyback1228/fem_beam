import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
def natural_spline(waypoints):
    dm_waypoints = ca.DM(waypoints)
    if dm_waypoints.shape[1]>3:
        dm_waypoints = dm_waypoints.T()

    dm_waypoints = dm_waypoints[:,0:2]
    n = dm_waypoints.shape[0]-1

    opti = ca.Opti()
    P1 = opti.variable(n,2)
    P2 = opti.variable(n,2)

    opti.subject_to(dm_waypoints[0,:]-2*P1[0,:]+P2[0,:]==0)
    opti.subject_to(P1[-1,:]-2*P2[-1,:]+dm_waypoints[n,:]==0)
    for i in range(1,n):
        opti.subject_to(P1[i,:]+P2[i-1,:]==2*dm_waypoints[i,:])
        opti.subject_to(P1[i-1,:]+2*P1[i,:]==2*P2[i-1,:]+P2[i,:])

    opti.solver('ipopt')
    sol = opti.solve()

    dm_p1 = sol.value(P1)
    dm_p2 = sol.value(P2)
    
    print("dm_waypoints")
    print(dm_waypoints)
    print("dm_p1")
    print(dm_p1)
    print("dm_p2")
    print(dm_p2)
    return dm_p1,dm_p2

def parametric_function(waypoints,p1,p2):
    dm_waypoints = ca.DM(waypoints)
    if dm_waypoints.shape[1]>3:
        dm_waypoints = dm_waypoints.T()

    dm_waypoints = dm_waypoints[:,0:2]

    mx_waypoints = ca.MX(ca.DM(waypoints))

    mx_p1 = ca.MX(ca.DM(p1))
    mx_p2 = ca.MX(ca.DM(p2))

    t = ca.MX.sym('t')
    n = dm_waypoints.shape[0]
    tau = ca.mod(t,n)
    i = ca.floor(tau)
    a = mx_waypoints[i,:]
    b = mx_p1[i,:]
    #print(b)
    c = mx_p2[i,:]
    i1 = ca.mod(i+1,n)
    d =mx_waypoints[i1,:]
    g = ca.power(1 - (tau-i), 3) * a + 3 * ca.power(1 - (tau-i), 2) * (tau-i) * b + 3 * (1 - (tau-i)) * ca.power(tau-i, 2) * c + ca.power(tau-i, 3) * d
    return ca.Function('f',[t],[g],['t'],['px'])


def direct_spline(waypoints,direct):
    p1_bar,p2_bar = natural_spline(waypoints)
    dm_waypoints = ca.DM(waypoints)
    if dm_waypoints.shape[1]>3:
        dm_waypoints = dm_waypoints.T()

    dm_waypoints = dm_waypoints[:,0:2]
    n = dm_waypoints.shape[0]-1

    opti = ca.Opti()
    P1 = opti.variable(n,2)
    P2 = opti.variable(n,2)

    dm_direct = ca.reshape(ca.DM(direct),(1,2))
    dm_direct = dm_direct/ca.norm_2(dm_direct)

    opti.subject_to((P1[0,:]-dm_waypoints[0,:])/ca.norm_2(P1[0,:]-dm_waypoints[0,:])==dm_direct)
    opti.subject_to(P1[-1,:]-2*P2[-1,:]+dm_waypoints[-1,:]==0)

    for i in range(1,n):
        opti.subject_to(P1[i,:]+P2[i-1,:]==2*dm_waypoints[i,:])
        opti.subject_to(P1[i-1,:]+2*P1[i,:]==2*P2[i-1,:]+P2[i,:])

    alpha = 1.0
    beta = 1000
    dp1 = P1-p1_bar
    dp2 = P2-p2_bar
    obj2 = ca.dot(dp1[:,0],dp1[:,0])+ca.dot(dp1[:,1],dp1[:,1])+ca.dot(dp2[:,0],dp2[:,0])+ca.dot(dp2[:,1],dp2[:,1])
    opti.minimize(alpha*ca.dot(dm_waypoints[0,:]-2*P1[0,:]+P2[0,:],dm_waypoints[0,:]-2*P1[0,:]+P2[0,:])+beta*obj2)

    opti.set_initial(P1,p1_bar)
    opti.set_initial(P2,p2_bar)

    opti.solver('ipopt')
    sol = opti.solve()

    dm_p1 = sol.value(P1)
    dm_p2 = sol.value(P2)
    
    
    return dm_p1,dm_p2


if __name__=='__main__':
    waypoints = np.array([[1.0,1.0],[2.0,1.4],[3.0,2.0],[3.5,1.0],[4.0,1.0]])
    n = len(waypoints)
    p1_bar,p2_bar = natural_spline(waypoints)
    f_nature = parametric_function(waypoints,p1_bar,p2_bar)

    p1_0,p2_0 = direct_spline(waypoints,[0,1.0])
    f_direct_0 = parametric_function(waypoints,p1_0,p2_0)

    p1_1,p2_1 = direct_spline(waypoints,[0.5,0.5])
    f_direct_1 = parametric_function(waypoints,p1_1,p2_1)

    p1_2,p2_2 = direct_spline(waypoints,[1.0,0.0])
    f_direct_2 = parametric_function(waypoints,p1_2,p2_2)

    t = ca.linspace(0,n-1-0.01,100).T
    l_nature = ca.reshape(f_nature(t),(2,-1)).T
    l_d0 = ca.reshape(f_direct_0(t),(2,-1)).T
    l_d1 = ca.reshape(f_direct_1(t),(2,-1)).T
    l_d2 = ca.reshape(f_direct_2(t),(2,-1)).T

    plt.scatter(waypoints[:,0],waypoints[:,1],color = '#0000ff',s=80,label='fitting points')
    plt.plot(l_nature[:,0],l_nature[:,1],label='nature spline')
    plt.plot(l_d0[:,0],l_d0[:,1],'--',label='initial direction = 90$^\circ$')
    plt.plot(l_d1[:,0],l_d1[:,1],'-*',label='initial direction = 45$^\circ$',markevery=5)
    plt.plot(l_d2[:,0],l_d2[:,1],'-o',label='initial direction = 0$^\circ$',markevery=5)

    plt.legend( fontsize="11",loc ="upper left")
    plt.show()


