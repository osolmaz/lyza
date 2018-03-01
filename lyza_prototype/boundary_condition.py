def join_boundaries(boundaries):
    def result(x):
        test = [b(x) for b in boundaries]
        return True in test
    return  result

class DirichletBC():
    def __init__(self, value, position_bool, components=None):
        self.value = value
        self.position_bool = position_bool
        self.components = components
