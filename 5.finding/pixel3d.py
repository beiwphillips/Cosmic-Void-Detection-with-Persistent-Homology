class Pixel3D:

    def __init__(self, x, y, z, v):
        self._x = x
        self._y = y
        self._z = z
        self._v = v
    def __lt__(self, other):
        return self._v < other._v
    def __gt__(self, other):
        return self._v > other._v
    def __repr__(self):
        return 'Pixel: [coordinates: ({}, {}, {}); value: {}]'.format(self._x, self._y, self._z, self._v)
    
    def getX(self):
        return self._x
    def setX(self, x):
        self._x = x
    def getY(self):
        return self._y
    def setY(self, y):
        self._y = y
    def getZ(self):
        return self._z
    def setZ(self, y):
        self._z = z
    def getV(self):
        return self._v
    def setV(self):
        self._v = v
    def getCoordinates(self):
        return (self._x, self._y, self._z)