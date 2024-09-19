#ifndef LOOP_EXPANSION_LIBRARY_H
#define LOOP_EXPANSION_LIBRARY_H
#include <vector>
#include <memory>
#include <map>
#include <set>
using namespace std;

namespace binsim{
    struct DFSInfo{
        int enterTime;
        int exitTime;
        int father;
        DFSInfo():enterTime(-1), exitTime(-1), father(-1){}
    };

    struct Edge{
        int src;
        int dst;
        Edge(int src, int dst):src(src), dst(dst){}
        Edge() = default;
    };

    class Graph{
    public:
        Graph();
        Graph(int nodeNum, const vector<int>&src, const vector<int>&dst, int entryNode);
        Graph(int nodeNum, const vector<vector<int>>& adjList, int entryNode);
        Graph(const Graph& graph) = default;
        ~Graph() = default;
        [[nodiscard]] int getNodeNum() const;
        [[nodiscard]] int getEntryNode() const;
        void checkAcyclicReducible();
        [[nodiscard]] bool isAcyclic();
        [[nodiscard]] bool isReducible();
        void addEdge(int src, int dst);
        void addEdges(const vector<int>&src, const vector<int>&dst);
        void toDAG(vector<int>& nodeId, vector<int>& edgeSrc, vector<int>& edgeDst, int k=0);
        [[nodiscard]] vector<int> findStronglyConnectedComponents() const;
        [[nodiscard]] vector<set<int>> findStronglyConnectedComponentsOnSubset(const set<int>& subset) const;
        [[nodiscard]] vector<int> calcDominatorTree() const;
        [[nodiscard]] pair<Graph, vector<set<int>>> reduce() const;
        void display();
        void toReducible();
    private:
        void dfsGraph(int cur, int& time,
                      vector<DFSInfo>& dfsInfo,
                      vector<Edge>& backEdges,
                      vector<Edge>& forwardEdges,
                      vector<vector<int> >& crossEdges,
                      vector<int>& selfLoop) const;
        void dfsOnSubset(vector<bool>& visited, vector<int>& exitTime, const vector<bool> &subsetFlags, int & time) const;
        void internalSCCOnSubset(vector<int> &sccId, const vector<bool>& subsetFlags) const;
        void internalSCC(vector<int> &sccId,
                         vector<Edge>& backEdges,
                         vector<int>& selfLoop,
                         vector<DFSInfo>& dfsInfo) const;
        int splitSingleNode(int node);
        void splitMultiNode(const set<int>& nodes, const vector<int>& original2new);
        void calcDFSPostOrder(int node, vector<int>& record, vector<bool>& visited) const;
        [[nodiscard]] vector<int> selectNodes() const;
        int nodeNum;
        int entryNode;
        vector<vector<int>> adjList;
        vector<vector<int>> adjListRev;
        vector<int> nodeTags;
        bool acyclic;   // whether current graph is acyclic.
        bool reducible; // whether current graph is reducible.
        bool modified;  // whether the graph has been modified, since last calculation for acyclic and reducible.
    };
}
#endif //LOOP_EXPANSION_LIBRARY_H
