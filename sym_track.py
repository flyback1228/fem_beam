import numpy as np
import casadi
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# returns the general Bezier cubic formula given 4 control points
def getCubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

def getCubicSym(i,a,b,c,d):
    t = casadi.MX.sym('t',1)
    
    g = casadi.power(1 - (t-i), 3) * a + 3 * casadi.power(1 - (t-i), 2) * (t-i) * b + 3 * (1 - (t-i)) * casadi.power(t-i, 2) * c + casadi.power(t-i, 3) * d
    return casadi.Function('f',[t], [g])

def getBezierCoef(waypoints):
    #interpolates with cubic bezier curves with cyclic boundary condition
    n = len(waypoints)
    M = np.zeros([n,n])
    #build M
    tridiagel = np.matrix([[1, 4, 1]])
    for idx in range(n-2):
        M[idx+1:idx+2, idx:idx+3] = tridiagel
    M[0,0:2]= tridiagel[:,1:3]
    M[-1,-2:]= tridiagel[:,0:2]
    M[0,-1] = tridiagel[:,0].reshape(1,-1)
    M[-1,0] = tridiagel[:,0].reshape(1,-1)
    #build sol vector
    s =np.zeros([n,2])
    for idx in range(n-1):
        s[idx,:] = 2*(2*waypoints[idx,:] + waypoints[idx+1,:])
    s[-1:] = 2*(2*waypoints[-1,:] + waypoints[0,:])

    #solve for a & b
    Ax = np.linalg.solve(M,s[:,0])
    Ay = np.linalg.solve(M,s[:,1])

    a = np.vstack([Ax,Ay])
    b = np.zeros([2,n])

    b[:,:-1] = 2*waypoints.T[:,1:] - a[:,1:]
    b[:,-1] = 2*waypoints.T[:,0] - a[:,0]
    
    return a.T, b.T

# return one cubic curve for each consecutive points
def getBezierCubic(waypoints):
    A, B = getBezierCoef(waypoints)
    length = len(waypoints)
    funs = [
        getCubic(waypoints[i], A[i], B[i], waypoints[i + 1])
        for i in range(length - 1)
    ]
    funs.append(getCubic(waypoints[length-1], A[length-1], B[length-1], waypoints[0]))
    return funs


def getBezierCubicSym(waypoints):   
    A, B = getBezierCoef(waypoints) 
    length = len(waypoints)    
    funs = [
        getCubicSym(i,waypoints[i], A[i], B[i], waypoints[i + 1])
        for i in range(length-1 )
    ]
    funs.append(getCubicSym(length-1,waypoints[length-1], A[length-1], B[length-1], waypoints[0]))
    return funs

def getSymbolicFunction(funs,t):   
    index = int(casadi.floor(t))
    return funs[index]

def parametricFunction(waypoints):
    #new_points = np.copy(waypoints)
    #new_points = np.append(waypoints,[waypoints[0,:]], axis=0)
    dm_waypoints = casadi.MX(casadi.DM(waypoints))
    
    #print(dm_waypoints)
    A, B = getBezierCoef(waypoints)  
    dm_A = casadi.MX(casadi.DM(A))
    dm_B = casadi.MX(casadi.DM(B))
       
    t = casadi.MX.sym('t')
    #t = t - casadi.floor(t/len(waypoints))
    
    #print(t)
    tau = casadi.mod(t,len(waypoints))
    i = casadi.floor(tau)
    a = dm_waypoints[i,:]
    #print(a)
    #print(dm_waypoints[2])
    b = dm_A[i,:]
    #print(b)
    c = dm_B[i,:]
    i1 = casadi.mod(i+1,len(waypoints))
    d =dm_waypoints[i1,:]
    g = casadi.power(1 - (tau-i), 3) * a + 3 * casadi.power(1 - (tau-i), 2) * (tau-i) * b + 3 * (1 - (tau-i)) * casadi.power(tau-i, 2) * c + casadi.power(tau-i, 3) * d
    return casadi.Function('f',[t],[g],['t'],['px'])

def parametricS(f,n):
    jac = f.jacobian() 
    t = casadi.MX.sym('t')
    s = casadi.MX.sym('s')
    dae={'x':s, 't':t, 'ode':casadi.norm_2(jac(t,0))}
    ts = 13
    integ = casadi.integrator('inte','cvodes',dae,{'grid':12})
    return integ   

class SymbolicTrack:
    def __init__(self,waypoints,width):
        resolution = 100
        #decimal = int(np.log10(resolution))


        # self.waypoints = np.genfromtxt(filename, delimiter=',')
        self.waypoints = waypoints
        self.width = width
        n = len(self.waypoints)
        self.max_t = n
        self.pt_t = parametricFunction(self.waypoints)    
        ts = np.linspace(0, n, n*resolution,endpoint=False)
        #print(ts.shape)
        self.center_line = np.array([self.pt_t(t)[0,:] for t in ts])
        self.center_line = np.reshape(self.center_line,(len(self.center_line),2))
        self.dsdt = self.pt_t.jacobian() 
        t = casadi.MX.sym('t')
        s = casadi.MX.sym('s')
        dae={'x':s, 't':t, 'ode':casadi.norm_2(self.dsdt(t,0))}
        integ = casadi.integrator('inte','cvodes',dae,{'grid':ts,'output_t0':True})
        s_inte = integ(x0=0)
        self.s_value = np.array(s_inte['xf'].T)
        self.s_value = np.reshape(self.s_value,len(self.s_value))
        #print(self.s_value.shape)

        theta = np.deg2rad(90)
        rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        self.inner_line = np.zeros_like(self.center_line)
        self.outer_line = np.zeros_like(self.center_line)

        for i in range(len(ts)):
            vec = self.dsdt(ts[i],0.0)
            vec = vec/np.linalg.norm(vec)
            vec = np.dot(rot_mat,vec)
            self.inner_line[i,:] = self.center_line[i,:]+width/2*vec.T
            self.outer_line[i,:] = self.center_line[i,:]-width/2*vec.T 

        self.s_to_t_lookup = casadi.interpolant("s_to_t","linear",[self.s_value.tolist()],ts.tolist())
        self.t_to_s_lookup = casadi.interpolant("t_to_s","linear",[ts.tolist()],self.s_value.tolist())

        self.max_s = self.s_value[-1]

        pt_t_mx = self.pt_t(t)
        jac = casadi.jacobian(pt_t_mx,t)
        hes = casadi.jacobian(jac,t)
        kappa = (jac[0]*hes[1]-jac[1]*hes[0])/casadi.power(casadi.norm_2(jac),3)
        self.f_kappa = casadi.Function('kappa',[t],[kappa])

        """
        cubic_funs = getBezierCubic(self.waypoints)
        cubic_funs_sym = getBezierCubicSym(self.waypoints)
        
        
        dev_funs = [f.jacobian() for f in cubic_funs_sym ]
        #self.center_line = np.array([fun(t) for fun in cubic_funs for t in np.linspace(0, 1, resolution,endpoint=False)])
        #self.center_line = np.array([getSymbolicFunction(cubic_funs_sym,t)(t) for t in np.linspace(0, n, n*resolution,endpoint=False)])
        #phi_vec = np.array([fun(t,0.0) for fun in dev_funs for t in np.linspace(0, 1, resolution,endpoint=False)])
        print(self.center_line.shape)
        phi_vec = np.array([getSymbolicFunction(dev_funs,t)(t,0.0) for t in np.linspace(0, n, n*resolution,endpoint=False)])
        
        phi = np.arctan2(phi_vec[:,1],phi_vec[:,0])[:,0]
        d = self.center_line[1:,:] - self.center_line[0:-1,:]
        #print(d)
        ds = np.linalg.norm(d,ord=2,axis=1)        
        ds = np.insert(ds,0,0.0)
        s = np.cumsum(ds)
        #print(s)
        self.smax = s[-1]
        #print(s.shape)
        #for i in range(0,len(s)-1):
        #    if(s[i+1]-s[i])<=0:
        #        print('not increasing at' +str(i))
        
        self.inner_waypoints = np.zeros((n,2))
        self.outer_waypoints = np.zeros((n,2))
        for i in range(0,n):
            vec = dev_funs[i](0.0,0.0)
            vec = vec/np.linalg.norm(vec)
            vec = np.dot(rot_mat,vec)
            #print(vec)
            #print(vec.shape)
            self.inner_waypoints[i,:] = self.waypoints[i,:]+width/2*vec.T
            self.outer_waypoints[i,:] = self.waypoints[i,:]-width/2*vec.T      
        
        inner_funs = getBezierCubic(self.inner_waypoints)
        outer_funs = getBezierCubic(self.outer_waypoints)
        self.inner_line = np.array([fun(t) for fun in inner_funs for t in np.linspace(0, 1, resolution,endpoint=False)])
        self.outer_line = np.array([fun(t) for fun in outer_funs for t in np.linspace(0, 1, resolution,endpoint=False)])
        
        
        
        bounds = np.array([np.min(self.outer_line[:,0]),np.max(self.outer_line[:,0]),np.min(self.outer_line[:,1]),np.max(self.outer_line[:,1])])
        bounds = np.round(bounds,decimal)
        #print(bounds)
        x_margin = np.round((bounds[1]-bounds[0])/20,decimal)
        y_margin = np.round((bounds[3]-bounds[2])/20,decimal)
        bounds = bounds + np.array([-x_margin,x_margin,-y_margin,y_margin])
        #print(bounds)
        self.x_axis = np.arange(bounds[0],bounds[1],1.0/resolution)
        self.y_axis = np.arange(bounds[2],bounds[3],1.0/resolution)    
        self.cost_map = -1*np.ones([len(self.x_axis),len(self.y_axis)])
        self.s_map = -1*np.ones([len(self.x_axis),len(self.y_axis)])
        self.phi_map = -1*np.ones([len(self.x_axis),len(self.y_axis)])
        T = KDTree(self.center_line)       
        
        for i in range(0,len(self.x_axis)):
            for j in range(0,len(self.y_axis)):
                dist,index = T.query((self.x_axis[i],self.y_axis[j]))
                if dist<=self.width/2:
                    self.cost_map[i,j] = 1-dist/(self.width/2)
                    self.s_map[i,j] = s[index]
                    self.phi_map[i,j] = phi[index]
        
        cost_value = np.zeros((2,len(self.x_axis),len(self.y_axis)))
        cost_value[0,:,:]=self.s_map
        cost_value[1,:,:]=self.cost_map
        s_value = np.zeros((len(s),3))
        s_value[:,0] = phi
        s_value[:,1:3] = self.center_line
        print(s[10])
        print(phi[10])
        print(self.center_line[10,:])
        print(s_value[10,:])
        self.phi_lookup = casadi.interpolant("phi_look","linear",[s],s_value.flatten())
        self.cost_lookup = casadi.interpolant('s_look','linear',[self.x_axis,self.y_axis],cost_value.flatten('F'))
        """
        pass

    def getPhiFromT(self,t):
        vec = self.dsdt(t,0.0)
        return np.arctan2(vec[1],vec[0])
    
    def getPhiSym(self,t):
        vec = self.dsdt(t,0.0)
        return casadi.arctan2(vec[1],vec[0])

    def getTangentVec(self,t):
        return self.dsdt(t,0.0)

    def getTFromS(self,s):
        return self.s_to_t_lookup(s)
    
    def getSFromT(self,t):
        return self.t_to_s_lookup(t)
    
    def getCurvatureSym(self,t):        
        fun = self.pt_t(t)
        jac = casadi.jacobian(fun,t)
        hes = casadi.jacobian(jac,t)
        return (jac[0]*hes[1]-jac[1]*hes[0])/casadi.power(casadi.norm_2(jac),3)

    def convertParameterToPos(self,t,n):
        #center = self.pt_t(t)        
        theta = np.deg2rad(90)
        rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        pt = np.zeros((len(t),2))
        for i in range(len(t)):
            vec = self.dsdt(t[i],0.0)
            vec = vec/np.linalg.norm(vec)
            vec = np.dot(rot_mat,vec)
            pt[i,:] = self.pt_t(t[i])+n[i]*vec.T
        return pt

    def plot(self,ax):
        #plt.pcolormesh(self.x_axis,self.y_axis,self.cost_map.T,figure=figure)
        #print(self.center_line.shape)
        ax.plot(self.center_line[:,0],self.center_line[:,1],'--b')
        ax.plot(self.inner_line[:,0],self.inner_line[:,1],'-g')
        ax.plot(self.outer_line[:,0],self.outer_line[:,1],'-y')
        ax.plot(self.waypoints[:,0],self.waypoints[:,1],'*r')
        #plt.plot(self.inner_waypoints[:,0],self.inner_waypoints[:,1],'*g',figure=figure)
        #plt.plot(self.outer_waypoints[:,0],self.outer_waypoints[:,1],'*y',figure=figure)

        #for i in range(0,len(self.waypoints)):
            #plt.arrow(self.waypoints[i,0],self.waypoints[i,1],float(self.vecs[i][0]),float(self.vecs[i][1]),figure = figure)




if __name__ == '__main__':
    waypoint_file = 'result/simulation/it_80.txt'
    angle = np.deg2rad(-170)
    direct = [np.cos(angle),np.sin(angle)]
    waypoints = casadi.DM.from_file(waypoint_file)
    waypoints = np.array(waypoints)
    
    my_track = SymbolicTrack(waypoints,0.25)
    fig = plt.subplot(1,1,1)
    my_track.plot(fig)
    
    t = casadi.MX.sym('t')
    f = casadi.Function('f',[t],[my_track.getCurvatureSym(t)])


    n = 14
    resolution = 100
    ts = np.linspace(0, n, n*resolution,endpoint=False)
    kappa = np.array([f(x) for x in ts]).reshape(len(ts))
    #kappa = np.reshape(kappa,len(ts))
    fig2 = plt.figure()
    plt.plot(ts,kappa)
    plt.grid()
    
    print(my_track.pt_t(ts))
    
    phi = np.array([my_track.getPhiFromT(t) for t in ts]).reshape(len(ts))
    fig3 = plt.figure()
    plt.plot(ts,phi)
    
    v = np.array([np.linalg.norm(my_track.getTangentVec(t)) for t in ts]).reshape(len(ts))
    
    plt.plot(ts,v)
    #angle = np.linspace(-4*np.pi,4*np.pi,100)
    #at = np.(angle, 1)
    #fig4 = plt.figure()
    #plt.plot(angle,at)
    
    
    plt.show()
    
    
    
    '''
    waypoints = np.genfromtxt('tracks/simpleoval.csv', delimiter=',')
    
    resolution = 100    
    n = len(waypoints)
    
    f = parametricFunction(waypoints)    
    ts = np.linspace(0, n, n*resolution,endpoint=False)
    print(ts.shape)
    center_line = np.array([f(t)[0,:] for t in ts])
    
    
    d = center_line[1:,:] - center_line[0:-1,:]
    d = np.reshape(d,(len(d),2))
    print(d.shape)
    ds = np.linalg.norm(d,ord=2,axis=1)        
    ds = np.insert(ds,0,0.0)
    
    s_num = np.cumsum(ds)
    print(s_num.shape)
    #print(s)
    #smax = s[-1]
    #print(f)
    #print(f(1.5))
    #print(f(13.999))
    
    jac = f.jacobian() 
    t = casadi.MX.sym('t')
    s = casadi.MX.sym('s')
    dae={'x':s, 't':t, 'ode':casadi.norm_2(jac(t,0))}
    integ = casadi.integrator('inte','cvodes',dae,{'grid':ts,'output_t0':True})
    s_inte = integ(x0=0)
    s_value = np.array(s_inte['xf'].T)
    #s_value = np.reshape(s_value,len(s_value))
    #s_value = np.insert(s_value,0,0.0)
    print(s_value)
    fig1 = plt.figure()
    plt.plot(ts,s_num)
    plt.plot(ts,s_value,'--r')
    fig2 = plt.figure()
    
    
    plt.show()
      
    
    #print(casadi.gradient(f,t))
    pos = f.jacobian()(13.9999,10)
    norm =casadi.norm_2(pos)
    print(pos)
    print(norm)
    print('--------')
    
    p = casadi.MX.sym('p')
    fun = f(t)
    jac = casadi.jacobian(fun,t)
    hes = casadi.jacobian(jac,t)
    print(hes)
    print(hes.size())
    #print(hes.jacobian())
    
    
    #integ = parametricS(f,14)
    #s = integ(x0=0)
    #print(s['xf'])
    '''