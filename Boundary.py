from copyreg import constructor
from typing import overload


class Boundary:
    def __init__(self,dx=False,dy=False,rz=False) -> None:
        self.dx=dx
        self.dy=dy
        self.rz=rz

    @staticmethod
    def create_pin():
        return Boundary(True,True,False)

    @staticmethod
    def create_fix():
        return Boundary(True,True,True)

    def __add__(self,other):
        dx=self.dx or other.dx
        dy=self.dy or other.dy
        rz=self.rz or other.rz
        return Boundary(dx,dy,rz)

    def __iadd__(self,other):
        self.dx |= other.dx
        self.dy |= other.dy
        self.rz |= other.rz
        return self
