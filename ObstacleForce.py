import shapely
from Element import *
import Force
import matplotlib.pyplot as plt

class ObstacleForce:
    def __init__(self, data):
        """Obstacle force applied to car

        Args:
            data (list like): store the obstacles
            k (double): potential field strength. Defaults to 2e6.
        """
        super().__init__()
        self.polygons=[]
        self.linerings=[]
        self.index_of_polygons=[]
        # i=int(0)
        # print(len(data))
        # self.points=[]
        for i,d in enumerate(data):
            d = np.reshape(d,(-1,2))
            polygon = shapely.Polygon(d)
            # for i in range(0,len(d)-1):
            #     pt = shapely.Point(d[i])
            #     self.points.append(pt)                
            self.polygons.append(polygon)
            self.linerings.append(polygon.exterior)
            self.index_of_polygons.append(i)
            for inter in polygon.interiors:
                self.linerings.append(inter)
                self.index_of_polygons.append(i)


        self.tree = shapely.STRtree(self.linerings)
        # self.tree = shapely.MultiPoint(self.points)
        # self.k = k

    def plot_obstacles(self):
        for d in self.linerings:
            x,y = d.xy
            plt.plot(x,y,'-g')
        plt.show()

    def apply_forces(self, node_list, k=2e6):
        
        pts = [shapely.Point(node.pos) for node in node_list]
        indice = self.tree.nearest(pts)

        for i,(idx,pt) in enumerate(zip(indice,pts)):
            p1,p2 = shapely.ops.nearest_points(pt,self.linerings[idx])

            direction = np.array([p1.x-p2.x,p1.y-p2.y,0.0])
            direction = (direction/np.linalg.norm(direction)).reshape(3,)
            # direction = direction*np.array([-tangents[i,1],tangents[i,0],0.0])
            
            polygon_index=self.index_of_polygons[idx]
            if(self.polygons[polygon_index].contains(p1)):
                f = -1e11*direction
            else:
                distance = shapely.distance(p1,p2)
                f = min(k/distance/distance,1e11)
            node_list[i].forces.append(Force(*f))
                

