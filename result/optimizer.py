
import casadi as ca
import yaml
import os

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

    def getMaxLongitudinalForce(self,Fz):
        return Fz*self.D_long

    def getMaxLateralForce(self,Fz):
        return Fz*self.D_lat

    def getAlphaAtMaxForce(self):
        return self.alpha_m

    def getLambdaAtMaxForce(self):
        return self.lambda_m
    

def optimize(track,N=50):    
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    tire_file = os.path.join(parent,'data/params/nwh_tire.yaml')
    params_file =os.path.join(parent,'data/params/racecar_nwh.yaml')
        
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

    st=track.get_max_s()
    s_array = ca.linspace(0,st,N+1).T()
    tau_array = track.s_to_t_lookup(s_array)
    tau_array_mid = track.s_to_t_lookup((s_array[0,0:-1]+s_array[0,1:])*0.5)
    kappa_array_mid = track.f_kappa(tau_array_mid);
    tangent_vec_array_mid= track.f_tangent_vec(tau_array_mid)

    phi_array_mid = ca.atan2(tangent_vec_array_mid[1,:],tangent_vec_array_mid[0,:])
    tangent_vec_norm_mid = ca.sqrt((tangent_vec_array_mid[0,:]*tangent_vec_array_mid[0,:]+tangent_vec_array_mid[1,:]*tangent_vec_array_mid[1,:]))
