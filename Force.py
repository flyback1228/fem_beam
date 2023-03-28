from numpy import float32


class Force:
    def __init__(self,fx=0.0,fy=0.0,mz=0.0) -> None:
        self.fx=float32(fx)
        self.fy=float32(fy)
        self.mz=float32(mz)

    def tolist(self):
        return [self.fx,self.fy,self.fz]


    def __add__(self,other):
        fx=self.fx+other.fx
        fy=self.fy+other.fy
        mz=self.mz+other.mz
        return Force(fx,fy,mz)


    def __iadd__(self,other):
        self.fx+=other.fx
        self.fy+=other.fy
        self.mz+=other.mz
        return self


        
        

        