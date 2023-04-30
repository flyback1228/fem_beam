from spline import direct_spline,parametric_function,natural_spline
import casadi as ca
import numpy as np

class Track():
    def __init__(self,waypoints, direct=None,resolution=100) -> None:
        if direct is None:
            p1,p2 = natural_spline(waypoints)
        else:
            p1,p2 = direct_spline(waypoints,direct)
        
        waypoints = ca.DM(waypoints)
        self.t_max = waypoints.rows()-1.0-0.00001
        self.pt_t = parametric_function(waypoints,p1,p2)
        
        ts = np.linspace(0.0,self.t_max,int(resolution*self.t_max))
        
        t = ca.MX.sym('t')
        n = ca.MX.sym('n')
        s = ca.MX.sym('s')
        
        pt_t_mx = self.pt_t(t)
        jac = ca.jacobian(pt_t_mx,t)
        hes = ca.jacobian(jac,t)
        
        self.f_tangent_vec = ca.Function("vec",[t],[jac]);
        kappa = (jac[0]*hes[1]-jac[1]*hes[0])/ca.power(ca.norm_2(jac),3)
        self.f_kappa = ca.Function('kappa',[t],[kappa])        
        self.f_phi = ca.Function('phi',[t],[ca.atan2(jac[1],jac[0])])
        
        dsdt = self.pt_t.jacobian() 
        dae={'x':s, 't':t, 'ode':ca.norm_2(dsdt(t,0))}
        integ = ca.integrator('inte','cvodes',dae,{'grid':ts,'output_t0':True})
        s_inte = integ(x0=0)
        self.s_value = np.array(s_inte['xf'].T)
        self.s_value = np.reshape(self.s_value,(len(self.s_value),))
        self.s_max=self.s_value[-1]
        # for i in range(len(ts)):
        #     vec = self.dsdt(ts[i],0.0)
        #     vec = vec/np.linalg.norm(vec)
        #     vec = np.dot(rot_mat,vec)
        #     self.inner_line[i,:] = self.center_line[i,:]+width/2*vec.T
        #     self.outer_line[i,:] = self.center_line[i,:]-width/2*vec.T 

        self.s_to_t_lookup = ca.interpolant("s_to_t","linear",[self.s_value.tolist()],ts.tolist())
        self.t_to_s_lookup = ca.interpolant("t_to_s","linear",[ts.tolist()],self.s_value.tolist())

        theta = np.deg2rad(90)
        rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
        xy =pt_t_mx.T + ca.mtimes(rot_mat,jac/ca.norm_2(jac))*n;

        self.f_tn_to_xy = ca.Function("tn_to_xy",[t,n],[xy]);
        
if __name__=='__main__':
    waypoint_file = 'result/simulation/it_80.txt'
    angle = np.deg2rad(-170)
    direct = [np.cos(angle),np.sin(angle)]
    waypoints = ca.DM.from_file(waypoint_file)
    track = Track(waypoints,direct)
    
    ts = ca.DM(np.linspace(0,waypoints.rows()-1-0.0001,100)).T
    print(np.reshape(track.pt_t(ts),(-1,2)))
    