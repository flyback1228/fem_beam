class Force:
    def __init__(self,fx=0.0,fy=0.0,mz=0.0) -> None:
        self.fx=fx
        self.fy=fy
        self.mz=mz

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


        
        

        