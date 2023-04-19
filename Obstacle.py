import shapely
import shapely.ops
import numpy as np
from shapely.geometry import shape
import matplotlib.pyplot as plt

class Obstacle():
    def __init__(self,data):
        """Obstacle force applied to car

        Args:
            data (list like): store the obstacles
            k (double): potential field strength. Defaults to 2e6.
        """
        self.polygons = []
        for d in data:
            poly = shapely.Polygon(np.reshape(d,(-1,2)))
            if not poly.contains(shapely.Point(0.0,0.0)):
                self.polygons.append(poly)
        #self.polygons = [ for d in data]
        self.multi_polygons = shapely.MultiPolygon(self.polygons)


    def get_neareset_points(self, state):
        pts = [shapely.Point(s[0],s[1]) for s in state]
        nearest_pts =  shapely.ops.nearest_points(self.multi_polygons,pts)
        #nearest_pts = nearest_pts[0]
        #print(nearest_pts)
        # npts = 
        #print(npts)
        nearest = [[pt.x,pt.y] for pt in nearest_pts[0]]
        return np.array(nearest)

    def plot_obstacles(self,ax=None):
        if ax is None:
            for d in self.polygons:
                x,y = d.exterior.xy
                plt.plot(x,y)
            plt.show()
        else:
            for d in self.polygons:
                x,y = d.exterior.xy
                ax.plot(x,y,'-g')