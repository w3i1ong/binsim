cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        pass

cdef extern from "<utility>" namespace "std":
    cdef cppclass pair[T1, T2]:
        pass

cdef extern from "<set>" namespace "std":
    cdef cppclass set[T]:
        pass

cdef extern from "UnionFindSet.cpp":
    pass

cdef extern from "UnionFindSet.h":
    cdef cppclass UnionFindSet:
        pass

cdef extern from "graph.cpp":
    pass

cdef extern from "graph.h" namespace "binsim":
    cdef cppclass Graph:
        Graph(int nodeNum, vector[int]&src, vector[int]&dst, int entry)
        Graph(Graph graph)
        int getNodeNum()
        int getEntryNode()
        void checkAcyclicReducible()
        bint isAcyclic()
        bint isReducible()
        void addEdge(int src, int dst)
        void addEdges(vector[int]&src, vector[int]&dst)
        vector[int] findStronglyConnectedComponents()
        void toDAG(vector[int]& nodeId, vector[int]&edgeSrc, vector[int]&edgeDst, int k)
        vector[int] calcDominatorTree()
        pair[Graph, vector[set[int]]] reduce()

