class Node:
    def __init__(self,id,pos) -> None:
        self.id = id
        self.pos=pos
        self.boundaries=[]
        self.forces=[]

    def add_force(self,force):
        self.forces.append(force)

    def add_boundary(self,boundary):
        self.boundaries.append(boundary)


