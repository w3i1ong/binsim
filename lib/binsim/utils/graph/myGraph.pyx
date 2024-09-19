# distutils: language=c++
from .graph cimport Graph, set, vector
from libcpp.pair cimport pair

cdef class pyGraph:
    cdef Graph *objPtr

    def __init__(self, nodeNum, U, V, entry):
        self.objPtr = new Graph(nodeNum, U, V, entry)

    def isReducible(self):
        return self.objPtr.isReducible()

    def isAcyclic(self):
        return self.objPtr.isAcyclic()

    def getNodeNum(self):
        return self.objPtr.getNodeNum()

    def __del__(self):
        if self.objPtr != NULL:
            del self.objPtr
            self.objPtr = NULL

    def findStronglyConnectedComponents(self):
        return self.objPtr.findStronglyConnectedComponents()

    def toDAG(self, k:int=0):
        cdef vector[int] nodeId, src, dst
        self.objPtr.toDAG(nodeId, src, dst, k)
        return nodeId, (src, dst)

    def findStronglyConnectedComponents(self):
        return self.objPtr.findStronglyConnectedComponents()

    def dominatorTree(self):
        cdef vector[int] tree
        tree = self.objPtr.calcDominatorTree()
        return tree

    def reduce(self):
        cdef pair[Graph, vector[set[int]]] result
        result = self.objPtr.reduce()
        graph = pyGraph(0, [], [], 0)
        del graph.objPtr
        graph.objPtr = new Graph(result.first)
        return graph, result.second

    def __repr__(self):
        return f"<CFG: {self.getNodeNum()} nodes>"


