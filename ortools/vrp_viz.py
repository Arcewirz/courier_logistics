import folium
import numpy as np
import random


colors = [
    'red',
    'blue',
    'gray',
    'darkgreen',
    'darkred',
    'lightred',
    'orange',
    'beige',
    'green',
    'lightgreen',
    'darkblue',
    'lightblue',
    'purple',
    'darkpurple',
    'pink',
    'cadetblue',
    'lightgray',
    'black'
]


def random_color():
    """Generate random color in hex format"""
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


class VRPPlot():
    """Store and visualise vrp data"""
    
    def __init__(self,
                 coords: list[tuple[float]],
                 trails: list[list[int]]=[],
                 one_way = True) -> None:
        """Initialize vrp plot instance.

        Args:
            coords (list[tuple[float]]): List of coords in (latitude, longtitude) format.
            trails (list[int], optional): Trails generated by a solver. List of lists of indexes
                corresponding to the coords order. Defaults to [] (no trails).
        """
        self.coords = coords
        self.trails = trails
        self.one_way = one_way
        
        
    def _plot(self, one_way):
        coords_array = np.array(self.coords)
        center = coords_array.mean(axis=0).tolist()
        ne = coords_array.min(axis=0).tolist()
        sw = coords_array.max(axis=0).tolist()
        
        plot = folium.Map(location=center, 
                          tiles="Stamen Terrain")

        folium.Marker(
            location=self.coords[0],
            icon=folium.Icon(icon='play', color='green')
        ).add_to(plot)

        #for loc in self.coords[1:]:
        #    folium.Marker(loc, icon=folium.Icon(icon='glyphicon ', prefix='fa', color='black',icon_color='#FFFF00')) \
        #    .add_to(plot)
            
        for enum, trail in enumerate(self.trails):
            for loc in trail:
                folium.Marker(self.coords[loc], icon=folium.Icon(icon='circle', color=colors[enum], icon_color='#00000')) \
                .add_to(plot)            
            
        for enum, trail in enumerate(self.trails):
            folium.PolyLine([self.coords[index] for index in trail],
                            color = colors[enum]) \
            .add_to(plot)
        
        if one_way:
            folium.Marker(
                location=self.coords[-1],
                icon=folium.Icon(icon='stop', color='red')
            ).add_to(plot)
        
        plot.fit_bounds((sw, ne))
        
        folium.Marker(
            location=self.coords[0],
            icon=folium.Icon(icon='play', color='green')
        ).add_to(plot)
        
        return plot
        
    
    def show(self):
        plot = self._plot(self.one_way)
        return plot
    
    
    def save(self, path="vrp_map.html") -> None:
        plot = self._plot()
        plot.save(path)
