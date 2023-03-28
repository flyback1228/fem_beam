from Node import Node
import numpy as np

class Element:
    def __init__(self,id:int,n1:Node,n2:Node,E,A,I,rho) -> None:
        self.id=id
        self.node1=n1
        self.node2=n2
        self.E=E
        self.A=A
        self.I=I
        self.rho=rho
        phi=self.angle
        c=np.cos(phi)
        s=np.sin(phi)
        l = self.length
        self.T = np.matrix([ [c , s, 0, 0, 0, 0],
                            [-s, c, 0, 0, 0, 0],
                            [ 0, 0, 1, 0, 0, 0],
                            [ 0, 0, 0, c, s, 0],
                            [ 0, 0, 0,-s, c, 0],
                            [ 0, 0, 0, 0, 0, 1] ],dtype=np.float32)

        self.Ke = E/l*np.matrix([ [ A,           0,        0, -A,            0,        0],
                               [ 0, 12*(I/l**2),  6*(I/l),  0, -12*(I/l**2),  6*(I/l)],
                               [ 0,     6*(I/l),      4*I,  0,     -6*(I/l),      2*I],
                               [-A,           0,        0,  A,            0,        0],
                               [ 0,-12*(I/l**2), -6*(I/l),  0,  12*(I/l**2), -6*(I/l)],
                               [ 0,     6*(I/l),      2*I,  0,     -6*(I/l),      4*I] ],dtype=np.float32)

        
        self.Me = rho*A*l/420*np.matrix([[140,      0,      0,      70,     0,      0],
                                        [0,         156,    22*l,   0,      54,     -13*l],
                                        [0,         22*l,   4*l*l,  0,      13*l, -3*l*l],
                                        [70,        0,      0,      140,    0,      0],
                                        [0,         54,     13*l,   0,      156,    -22*l],
                                        [0,         -13*l,  -3*l*l, 0,      -22*l,4*l*l]],dtype=np.float32)

        
        self.Kg = self.T.transpose()*self.Ke*self.T
        self.Mg = self.T.transpose()*self.Me*self.T
        
        pass

    @property
    def length(self):
        return np.linalg.norm(self.node1.pos-self.node2.pos)

    @property
    def angle(self):
        diff = self.node2.pos - self.node1.pos
        return np.arctan2(diff[1],diff[0],dtype=np.float32)

    def local_stiffness_matrix(self):
        return self.Ke

    def global_stiffness_matrix(self):
        return self.Kg

    def local_mass_matrix(self):
        return self.Me

    def global_mass_matrix(self):
        return self.Mg

        
        

        