
class Domain:
    def is_subset(self, cell, is_boundary):
        raise Exception('Do not use the base class')

class DefaultDomain(Domain):
    def is_subset(self, cell):
        return not cell.is_boundary

class AllDomain(Domain):
    def is_subset(self, cell):
        return True


