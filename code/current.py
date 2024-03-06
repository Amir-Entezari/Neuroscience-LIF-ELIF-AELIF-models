from pymonntorch import *


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None, required=True)
        ng.I = ng.vector(self.value)
