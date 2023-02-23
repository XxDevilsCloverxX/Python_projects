# hw7.py
# Implementation of a simple Polygon class

class Polygon:

    def __init__(self):
        # Initialize a Polygon object.
        self.x = [] # self.x[i] is the x-coordinate of vertex 'i'.
        self.y = [] # self.y[i] is the y-coordinate of vertex 'i'.

    def set_vertices(self, coordinates):
        # Set the vertices for this Polygon object.
        # 'coordinates' is a list of floats, interpretted as follows: x0, y0, x1, y1...
        n = len(coordinates)
        if n%2 != 0:
            print("set_vertices()  Error: 'coordinates' must"+
                  "contain an even number of elements!")
            return False
        self.x = [coordinates[2*i] for i in range(n//2)]
        self.y = [coordinates[2*i+1] for i in range(n//2)]
        return True

    def num_vertices(self):
        # Return the number of vertices in this Polygon object.
        return len(self.x)

    def perimeter(self):
        # Return the perimeter of this Polygon object.

        perim = 0
        for i in range(self.num_vertices()):
            perim += ((self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2)**(1/2)
        return perim

    def area(self):
        # Return the area of this Polygon object.

        area = 0
        for i in range(self.num_vertices()):
            area += self.x[i-1]*self.y[i] - self.x[i]*self.y[i-1]
        return(abs(area) *1/2)

    def get_string(self):
        n = self.num_vertices()
        # Return a string representation of this Polygon object.
        # It will be a list of of the vertices, in order.
        result = "" # Start with empty string, and append each vertex to it.
        for i in range(n):
            result += "({0}, {1}) ".format(self.x[i], self.y[i])
        return result


P = Polygon() # Create a Polygon object, P.

# This next line sets 'P' to be the polygon
# whose vertices are (2,0), (1,1), (1,0).

P.set_vertices([2,0, 1,1, 1,0])
# Print out some information about the Polygon P.
print("P has vertices : "+P.get_string())
p = P.perimeter()
print("Its perimeter is {0}".format(p))
a = P.area()
print("Its area is {0}".format(a))
