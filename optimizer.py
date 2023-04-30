
import casadi as ca
import yaml
import os
import numpy as np
from track import Track

class SimplePacTireMode:
    def __init__(self,params)->None:
        self.B_long = params['B_long']       
        self.C_long = params['C_long']
        self.D_long = params['D_long']
        self.B_lat = params['B_lat']       
        self.C_lat = params['C_lat']
        self.D_lat = params['D_lat']

        self.lambda_m = ca.tan(ca.pi/(2*self.C_long))/self.B_long
        self.alpha_m = ca.tan(ca.pi/(2*self.C_lat))/self.B_lat
        pass

    def getForce(self,lamb,alpha,Fz):
        return ca.veccat(Fz*self.D_long*ca.sin(self.C_long*ca.atan(self.B_long*lamb)),
                                Fz*self.D_lat*ca.sin(self.C_lat*ca.atan(self.B_lat*alpha)))

    def getLongitudinalForce(self,lamb,Fz):
        return Fz*self.D_long*ca.sin(self.C_long*ca.atan(self.B_long*lamb))
    
    def getLateralForce(self,alpha,Fz):
        return Fz*self.D_lat*ca.sin(self.C_lat*ca.atan(self.B_lat*alpha))

    def getMaxLongitudinalForce(self,Fz):phi0=-170/180.0*ca.pi

    def getLambdaAtMaxForce(self):
        return self.lambda_m
    

def optimize(track,path_width,N=50): 
    
    nx = 7
    nu = 3
    
    IDX_X_t = 0
    IDX_X_n = 1
    IDX_X_phi = 2
    IDX_X_vx = 3
    IDX_X_vy= 4
    IDX_X_omega = 5
    IDX_X_delta = 6

    IDX_U_Ddelta = 0
    IDX_U_Throttle = 1
    IDX_U_Brake = 2

    phi0=-170/180.0*ca.pi
    v0=0.15
    X0 = ca.vertcat([0.0, 0, phi0, v0,0,0,0]);
    
    current = os.path.dirname(os.path.realpath(__file__))
    tire_file = os.path.join(current,'data/params/nwh_tire.yaml')
    params_file =os.path.join(current,'data/params/racecar_nwh.yaml')
        
    # define tire mode
    with open(tire_file) as file:
        front_tire_params = yaml.safe_load(file)
    front_tire_model = SimplePacTireMode(front_tire_params)

    with open(tire_file) as file:
        rear_tire_params = yaml.safe_load(file)
    rear_tire_model = SimplePacTireMode(rear_tire_params)

    # define model
    with open(params_file) as file:
        params = yaml.safe_load(file)

    mass = params['m']
    Iz = params["Iz"]
    cd = params["Cd"]
    cm0_ = params["Cm0"]
    cm1_ = params["Cm1"]
    cm2_ = params["Cm2"]
    cbf_ = params["Cbf"]
    cbr_ = params["Cbr"]

    lf = params["lf"]
    lr = params["lr"]


    v_min_ = params["v_min"]
    v_max_ = params["v_max"]
    d_min_ = params["d_min"]
    d_max_ = params["d_max"]
    delta_min = params["delta_min"]
    delta_max = params["delta_max"]
    delta_dot_min = params["delta_dot_min"]
    delta_dot_max = params["delta_dot_max"]

    st=track.s_max
    s_array = ca.linspace(0,st,N+1).T
    tau_array = track.s_to_t_lookup(s_array)
    tau_array_mid = track.s_to_t_lookup((s_array[0,0:-1]+s_array[0,1:])*0.5)
    kappa_array_mid = track.f_kappa(tau_array_mid);
    tangent_vec_array_mid= track.f_tangent_vec(tau_array_mid)

    phi_array_mid = ca.atan2(tangent_vec_array_mid[1,:],tangent_vec_array_mid[0,:])
    tangent_vec_norm_mid = ca.sqrt((tangent_vec_array_mid[0,:]*tangent_vec_array_mid[0,:]+tangent_vec_array_mid[1,:]*tangent_vec_array_mid[1,:]))


    opti = ca.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu, N)
    dt_sym_array = opti.variable(1,N)

    n_sym_array = (X[IDX_X_n,0:-1]+X[IDX_X_n,1:])*0.5
    phi_sym_array = (X[IDX_X_phi, 0:-1]+X[IDX_X_phi, 1:])*0.5
    vx_sym_array = (X[IDX_X_vx,0:-1]+X[IDX_X_vx,1:])*0.5
    vy_sym_array = (X[IDX_X_vy,0:-1]+X[IDX_X_vy,1:])*0.5
    omega_sym_array = (X[IDX_X_omega,0:-1]+X[IDX_X_omega,1:])*0.5
    delta_sym_array = (X[IDX_X_delta,0:-1]+X[IDX_X_delta,1:])*0.5

    delta_dot_sym_array = U[IDX_U_Ddelta,:]
    throttle_sym_array = U[IDX_U_Throttle,:]
    brake_sym_array = U[IDX_U_Brake,:]


    # n_sym_array = X(1,Slice());
    # n_obj = (casadi::MX::atan(5 * ( n_sym_array*n_sym_array - track->get_width()*track->get_width() / 4) ) + casadi::pi / 2) * 12;

    dphi_c_sym_array = phi_sym_array - phi_array_mid;
    fx_f_sym_array = cd * throttle_sym_array - cm0_ - cm1_ * vx_sym_array * vx_sym_array - cm2_ * vy_sym_array * vy_sym_array - cbf_ * brake_sym_array;
    fx_r_sym_array = cd * throttle_sym_array - cm0_ - cm1_ * vx_sym_array * vx_sym_array - cm2_ * vy_sym_array * vy_sym_array - cbr_ * brake_sym_array;

    alpha_f = ca.atan2(omega_sym_array * lf + vy_sym_array, -vx_sym_array) + delta_sym_array;
    alpha_r = ca.atan2(omega_sym_array * lr - vy_sym_array, vx_sym_array);

    Fz = ca.DM.ones(2) * mass / 2 * 9.81
    fy_f_sym_array = front_tire_model.getLateralForce(alpha_f, float(Fz[0]));
    fy_r_sym_array = rear_tire_model.getLateralForce(alpha_r, float(Fz[1]));
    
    n_obj = ca.exp(10*(X[IDX_X_n,:]*X[IDX_X_n,:]/(path_width*path_width/4)-1))

    opti.minimize(10*ca.sum2(dt_sym_array) + 1*ca.dot(delta_dot_sym_array,delta_dot_sym_array) + 10*ca.sum2(n_obj))

    opti.subject_to(X[0, 1:] == X[0, 0:-1] + dt_sym_array * (vx_sym_array * ca.cos(dphi_c_sym_array) - vy_sym_array * ca.sin(dphi_c_sym_array))/(tangent_vec_norm_mid*(1-n_sym_array*kappa_array_mid)))
    opti.subject_to(X[1, 1:] == X[1, 0:-1] + dt_sym_array * (vx_sym_array * ca.sin(dphi_c_sym_array) + vy_sym_array* ca.cos(dphi_c_sym_array)))
    opti.subject_to(X[2, 1:] == X[2, 0:-1] + dt_sym_array * omega_sym_array)
    opti.subject_to(X[3, 1:] == X[3, 0:-1] + dt_sym_array * (fx_r_sym_array + fx_f_sym_array * ca.cos(delta_sym_array) - fy_f_sym_array * ca.sin(delta_sym_array) + mass * vy_sym_array* omega_sym_array)/ mass)
    opti.subject_to(X[4, 1:] == X[4, 0:-1] + dt_sym_array * (fy_r_sym_array + fx_f_sym_array * ca.sin(delta_sym_array) + fy_f_sym_array * ca.cos(delta_sym_array) - mass * vx_sym_array * omega_sym_array)/ mass)
    opti.subject_to(X[5, 1:] == X[5, 0:-1] + dt_sym_array * (fy_f_sym_array * lf * ca.cos(delta_sym_array) + fx_f_sym_array * lf * ca.sin(delta_sym_array) - fy_r_sym_array * lr)/ Iz)
    opti.subject_to(X[6, 1:] == X[6, 0:-1] + dt_sym_array * delta_dot_sym_array)


    opti.subject_to(X[0, :] == tau_array);
    opti.subject_to(X[:, 0] == X0);
    opti.subject_to(dt_sym_array >0);
    opti.subject_to(X[IDX_X_vx, :] >0);


    opti.subject_to(opti.bounded(delta_min, X[IDX_X_delta,:], delta_max));

    opti.subject_to(opti.bounded(delta_dot_min, delta_dot_sym_array, delta_dot_max));
    opti.subject_to(opti.bounded(0, throttle_sym_array, d_max_));
    opti.subject_to(opti.bounded(0, brake_sym_array, 1));
    
    
    
    X_guess = ca.DM.zeros(nx,N+1);
    X_guess[:, 0] = X0;
    

    X_guess[IDX_X_phi,0] = phi0;
    phi_array = track.f_phi(tau_array);
    for i in range(1,N+1):
        X_guess[IDX_X_phi,i] = phi_array[i];
        if(float(phi_array[i])-float(X_guess[IDX_X_phi,i-1])<-ca.pi):
            X_guess[IDX_X_phi,i]=X_guess[IDX_X_phi,i]+2*ca.pi
        elif(float(phi_array[i])-float(X_guess[IDX_X_phi,i-1])>ca.pi):
            X_guess[IDX_X_phi,i]=X_guess[IDX_X_phi,i]-2*ca.pi
        
    

    X_guess[IDX_X_vx,:]=v0;
    X_guess[IDX_X_t,:] = tau_array;

    option = {}
    option["max_iter"] = 30000;
    option["tol"]=1e-6;
    option["linear_solver"]="ma57";

    opti.set_initial(X, X_guess);
    opti.set_initial(dt_sym_array,(s_array[1:]-s_array[0:-1])/v0);
    opti.solver("ipopt", {}, {});
    sol = opti.solve()
    return sol.value(dt_sym_array),sol.value(X),sol.value(U)
    
if __name__=='__main__':
    
    script_dir = os.path.dirname(__file__)
    phi0=-170/180.0*ca.pi
    direct = [np.cos(phi0),np.sin(phi0)]
    
    result_file = os.path.join(script_dir, 'result/simulation/it_80.txt')
    result = np.array(ca.DM.from_file(result_file))
    track = Track(result,direct)

    
    t,x,u = optimize(track,4.5)
    print(x)
    xy = track.f_tn_to_xy(ca.DM(x[0,:]).T,ca.DM(x[1,:]).T)
    print(xy)