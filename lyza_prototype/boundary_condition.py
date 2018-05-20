def join_boundaries(boundaries):
    def result(x, t):
        test = [b(x, t) for b in boundaries]
        return True in test
    return  result

class DirichletBC():
    def __init__(self, function, position_bool, components=None):
        self.function = function
        self.position_bool = position_bool
        self.components = components
        self.time = 0

    def set_time(self, t):
        self.time = t

    def value(self, coor):
        return self.function(coor, self.time)
