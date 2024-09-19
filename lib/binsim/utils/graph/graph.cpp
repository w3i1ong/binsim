#include "graph.h"
#include <algorithm>
#include <set>
#include <map>
#include <cassert>
#include <queue>
#include <iostream>
#include "UnionFindSet.h"
#include <utility>
using namespace std;

binsim::Graph::Graph() {
    this->nodeNum = 0;
    this->entryNode = 0;
    this->acyclic = true;
    this->reducible = true;
    this->modified = true;
}

binsim::Graph::Graph(int nodeNum, const vector<int> &src, const vector<int> &dst, int entryNode) {
    this->nodeNum = nodeNum;
    this->entryNode = entryNode;
    this->adjList.resize(nodeNum);
    this->adjListRev.resize(nodeNum);
    for(size_t i = 0; i < src.size(); i++){
        this->adjList[src[i]].push_back(dst[i]);
        this->adjListRev[dst[i]].push_back(src[i]);
    }
    for(int i = 0; i < this->getNodeNum(); i++){
        this->nodeTags.emplace_back(i);
    }
    this->modified = true;
    this->acyclic = true;
    this->reducible = true;
}

binsim::Graph::Graph(int nodeNum, const vector<vector<int>>& adjList, int entryNode) {
    this->nodeNum = nodeNum;
    this->entryNode = entryNode;
    this->adjList = adjList;
    this->adjListRev.resize(nodeNum);
    for(int i = 0; i< nodeNum; i++){
        for(auto &next: adjList[i]){
            this->adjListRev[next].push_back(i);
        }
    }
    this->nodeTags.reserve(this->getNodeNum());
    for(int i = 0; i < this->getNodeNum(); i++){
        this->nodeTags.emplace_back(i);
    }
    this->acyclic = true;
    this->reducible = true;
    this->modified = true;
}

int binsim::Graph::getNodeNum() const {
    return this->nodeNum;
}

int binsim::Graph::getEntryNode() const {
    return this->entryNode;
}

bool binsim::Graph::isAcyclic() {
    if(this->modified){
        this->checkAcyclicReducible();
        this->modified = false;
    }
    return this->acyclic;
}

bool binsim::Graph::isReducible() {
    if(this->modified){
        this->checkAcyclicReducible();
        this->modified = false;
    }
    return this->reducible;
}

inline void binsim::Graph::addEdge(int src, int dst) {
    this->adjList[src].push_back(dst);
    this->adjListRev[dst].push_back(src);
    this->modified = true;
}

inline void binsim::Graph::addEdges(const vector<int> &src, const vector<int> &dst) {
    for(size_t i = 0; i < src.size(); i++){
        this->adjList[src[i]].push_back(dst[i]);
        this->adjListRev[dst[i]].push_back(src[i]);
    }
    this->modified = true;
}

// Convert current Control Flow Graph in to Directed Acyclic Graph.
void binsim::Graph::toDAG(vector<int>& nodeId, vector<int>& edgeSrc, vector<int>& edgeDst, int k){
    edgeDst.clear(), edgeSrc.clear(), nodeId.clear();
    // if current graph is acyclic, just return it.
    if(this->isAcyclic()){
        for(int i = 0; i< this->nodeNum; i++){
            nodeId.emplace_back(i);
        }
        // As the graph is acyclic, the number of edges is at least nodeNum-1.
        edgeSrc.reserve(this->nodeNum), edgeDst.reserve(this->nodeNum);
        for(int i = 0; i< this->nodeNum; i++){
            for(auto &next: this->adjList[i]){
                edgeSrc.emplace_back(i);
                edgeDst.emplace_back(next);
            }
        }
        return;
    }
    // if the graph is not reducible, we have to convert it to reducible graph first;
    if(!this->isReducible()){
        this->toReducible();
        this->toDAG(nodeId, edgeSrc, edgeDst, k);
        for(auto& id: nodeId)
            id = this->nodeTags[id];
        return;
    }
    // find all the strongly connected components and back edges
    vector<int> sccId;
    vector<Edge> backEdges;
    vector<int> selfLoopNodes;
    vector<DFSInfo> dfsInfo(this->nodeNum);
    this->internalSCC(sccId, backEdges, selfLoopNodes, dfsInfo);
    vector<set<int> > components(*max_element(sccId.begin(), sccId.end())+1);
    for(int i = 0; i < this->nodeNum; i++){
        components[sccId[i]].insert(i);
    }

    // find all self-loops
    set<int> selfLoop(selfLoopNodes.begin(), selfLoopNodes.end());
    nodeId.reserve(this->nodeNum);
    for(int i = 0; i < this->nodeNum; i++){
        nodeId.push_back(i);
    }

    edgeDst.reserve(this->nodeNum), edgeSrc.reserve(this->nodeNum);
    for(int i = 0; i< this->nodeNum; i++){
        for(auto &next: this->adjList[i]){
            // skip all back-edges.
            // If the given graph is reducible, it is clear that,
            // a.enterTime >= b.enterTime and a.exitTime <= b.exitTime <=> (a,b) is a backward edge.
            if(dfsInfo[i].enterTime >= dfsInfo[next].enterTime && dfsInfo[i].exitTime <= dfsInfo[next].exitTime){
                continue;
            }
            edgeDst.push_back(next);
            edgeSrc.push_back(i);
        }
    }

    // if k == 0, we don't need to generate duplicates for nodes.
    if(k == 0)
        return;

    for(auto & scc: components){
        // If the component consists of only one node and it isn't in a self-loop, just skip it.
        // We will not duplicate it.
        if(scc.size() == 1 && selfLoop.find(*scc.begin()) == selfLoop.end()) {
            continue;
        }
        vector<Edge> sccInternalEdges, sccOutEdges, sccBackEdges;
        for(auto node: scc){
            for(auto &next: this->adjList[node]){
                if(scc.count(next)){
                    if(dfsInfo[node].enterTime >= dfsInfo[next].enterTime && dfsInfo[node].exitTime <= dfsInfo[next].exitTime)
                        sccBackEdges.emplace_back(node, next);
                    else
                        sccInternalEdges.emplace_back(node, next);
                }
                else{
                    sccOutEdges.emplace_back(node, next);
                }
            }
        }

        map<int,int> node2base;
        for(auto node: scc){
            node2base[node] = nodeId.size();
            for(int i = 0; i< k; i++)
                nodeId.push_back(node);
        }
        for(auto &edge: sccInternalEdges){
            for(int i = 0; i< k; i++){
                edgeSrc.push_back(node2base[edge.src]+i);
                edgeDst.push_back(node2base[edge.dst]+i);
            }

        }
        for(auto &edge: sccOutEdges){
            for(int i = 0; i< k; i++){
                edgeSrc.push_back(node2base[edge.src]+i);
                edgeDst.push_back(edge.dst);
            }
        }
        if(k>0) {
            for (auto &edge: sccBackEdges) {
                for (int i = 1; i < k; i++) {
                    edgeSrc.push_back(node2base[edge.src] + i - 1);
                    edgeDst.push_back(node2base[edge.dst] + i);
                }
                edgeSrc.push_back(edge.src);
                edgeDst.push_back(node2base[edge.dst]);
            }
        }
    }
}

void binsim::Graph::dfsGraph(int cur, int& time,
                             vector<DFSInfo>& dfsInfo,
                             vector<Edge>& backEdges,
                             vector<Edge>& forwardEdges,
                             vector<vector<int>>& crossEdges,
                             vector<int>& selfLoop) const {
    dfsInfo[cur].enterTime = time++;
    for(auto &next: this->adjList[cur]){
        // note: we ignore self-loops, as all self-loops will be removed in T1 operation
        if(next == cur) {
            selfLoop.push_back(cur);
            continue;
        }
        // if the node is not visited, dfs it
        if(dfsInfo[next].enterTime == -1){
            dfsGraph(next, time, dfsInfo, backEdges, forwardEdges, crossEdges, selfLoop);
            dfsInfo[next].father = cur;
        }
        else{
            // if the node is visited, check whether it is a back edge, forward edge or cross edge
            if(dfsInfo[next].exitTime == -1){
                // back edge
                backEdges.emplace_back(cur, next);
            }
            else if(dfsInfo[next].enterTime > dfsInfo[cur].enterTime){
                // forward edge
                forwardEdges.emplace_back(cur, next);
            }
            else if(dfsInfo[next].enterTime < dfsInfo[cur].enterTime){
                // cross edge
                crossEdges[next].push_back(cur);
            }
        }
    }
    dfsInfo[cur].exitTime = time++;
}

// check whether the cyclic graph is reducible
// It is based on the paper [Testing flow graph reducibility](https://dl.acm.org/doi/10.1145/800125.804040)
void binsim::Graph::checkAcyclicReducible() {
    vector<DFSInfo> dfsInfo(this->nodeNum);
    vector<Edge> backEdges, forwardEdges;
    vector<vector<int> > crossEdges(this->nodeNum);
    vector<int> selfLoop;
    int time = 0;
    dfsGraph(this->entryNode, time, dfsInfo,
             backEdges, forwardEdges, crossEdges, selfLoop);
    assert(time == 2 * this->adjList.size());

    // if there is no back edge, the graph is acyclic
    this->acyclic = backEdges.empty() && selfLoop.empty();

    // A DAG must be reducible
    if(this->acyclic){
        this->reducible = true;
        return;
    }
    // Check whether it is reducible
    vector<int> highPT = vector<int>(this->nodeNum, -1);
    sort(backEdges.begin(), backEdges.end(), [&dfsInfo](const Edge& a, const Edge& b){
        return dfsInfo[a.dst].enterTime > dfsInfo[b.dst].enterTime;
    });
    set<int> check;
    for(auto edge : backEdges){
        int src = edge.src, dst = edge.dst;
        check.insert(src);
        while(!check.empty()){
            int u = *check.begin();
            check.erase(check.begin());
            if(dfsInfo[dst].enterTime > dfsInfo[u].enterTime || dfsInfo[dst].exitTime < dfsInfo[u].exitTime){
                this->reducible = false;
                return;
            }
            while(u!=dst){
                if(highPT[u] == -1){
                    highPT[u] = dst;
                    for(auto prev : crossEdges[u]){
                        check.insert(prev);
                    }
                }
                u = dfsInfo[u].father;
            }
        }
    }
    for(auto& edge : forwardEdges){
        int u = edge.src, v = edge.dst;
        if(highPT[v] != -1 && dfsInfo[u].enterTime<dfsInfo[highPT[v]].enterTime) {
            this->reducible = false;
            return;
        }
    }
    this->reducible = true;
}


void dfsGraphWithExitTime(int node, int &time, const vector<vector<int>>& adjList,
                                 vector<bool>& visited,
                                 vector<int>& exitTime){
    visited[node] = true;
    for(auto &next: adjList[node]){
        if(!visited[next]){
            dfsGraphWithExitTime(next, time, adjList, visited, exitTime);
        }
    }
    exitTime[node] = time;
    time += 1;
}


vector<int> binsim::Graph::findStronglyConnectedComponents() const {
    vector<int> sccId(this->nodeNum, -1);
    vector<bool> visited(this->nodeNum, false);
    vector<int> exitTime(this->nodeNum, -1);
    int time = 0;
    dfsGraphWithExitTime(this->entryNode, time, this->adjList, visited, exitTime);
    vector<int> nodeOrderedByExitTime(this->nodeNum, -1);
    for(int i = 0; i< this->nodeNum; i++){
        nodeOrderedByExitTime[this->nodeNum - 1 - exitTime[i]] = i;
    }
    visited = vector<bool>(this->nodeNum, false);
    int curSccId = 0;
    for(auto &node: nodeOrderedByExitTime){
        if(sccId[node] == -1){
            queue<int> q;
            q.push(node);
            while(!q.empty()){
                int u = q.front();
                q.pop();
                sccId[u] = curSccId;
                visited[u] = true;
                for(auto &next: this->adjListRev[u]){
                    if(!visited[next]){
                        q.push(next);
                    }
                }
            }
        }
        curSccId += 1;
    }
    return sccId;
}

vector<set<int>> binsim::Graph::findStronglyConnectedComponentsOnSubset(const set<int>& subset) const {
    vector<int> sccId;
    vector<bool> subsetFlag(this->nodeNum, false);
    for(auto node: subset){
        subsetFlag[node] = true;
    }
    this->internalSCCOnSubset(sccId,subsetFlag);
    vector<set<int> > results;
    for(size_t i = 0; i< sccId.size(); i++){
        int id = sccId[i];
        if(id == -1){
            continue;
        }
        if(id >= (int)results.size()){
            results.resize(id+1);
        }
        results[id].insert((int)i);
    }
    for(size_t i = 0; i< results.size(); i++){
        if(results[i].size() == 1){
            swap(results[i], results.back());
            results.pop_back();
            i--;
        }
    }
    return results;
}

void binsim::Graph::dfsOnSubset(vector<bool> &visited, vector<int> &exitTime, const vector<bool> &subsetFlags, int & time) const {
    for(int i = 0; i < this->nodeNum; i++){
        if(!subsetFlags[i] || visited[i])
            continue;
        visited[i] = true;
        for(auto &next: this->adjList[i]){
            if(subsetFlags[next] && !visited[next]){
                dfsOnSubset(visited, exitTime, subsetFlags, time);
            }
        }
        exitTime[i] = time ++;
    }
}

void binsim::Graph::internalSCCOnSubset(vector<int> &sccId, const vector<bool> &subsetFlags) const {
    // step 1: dfs on subset and record exit time
    sccId.resize(this->nodeNum, -1);
    vector<int> exitTime(this->nodeNum, -1);
    vector<bool> visited(this->nodeNum, false);
    int time = 0;
    dfsOnSubset(visited, exitTime, subsetFlags, time);
    // step 2: sort nodes by exit time
    vector<int> sortedNodes(time);
    for(int i = 0; i< this->nodeNum; i++){
        if(subsetFlags[i]){
            sortedNodes[time - 1 - exitTime[i]] = i;
        }
    }
    // step 3: dfs on subset in reverse order
    visited = vector<bool>(this->nodeNum, false);
    int curSccId = 0;
    for(auto&node:sortedNodes){
        if(visited[node])
            continue;
        queue<int> q;
        q.push(node);
        while(!q.empty()){
            int u = q.front();
            q.pop();
            sccId[u] = curSccId;
            visited[u] = true;
            for(auto &next: this->adjListRev[u]){
                if(subsetFlags[next] && !visited[next]){
                    q.push(next);
                }
            }
        }
        curSccId++;
    }
}



void binsim::Graph::internalSCC(vector<int> &sccId,
                                vector<Edge>& backEdges,
                                vector<int>& selfLoop,
                                vector<DFSInfo>&dfsInfo) const{
    sccId.resize(this->nodeNum, -1);
    backEdges.resize(0);
    dfsInfo.resize(this->nodeNum);
    selfLoop.resize(0);
    vector<Edge> forwardEdges;
    vector<vector<int> > crossEdges(this->nodeNum);
    int time = 0;
    dfsGraph(this->entryNode, time, dfsInfo,
             backEdges, forwardEdges, crossEdges, selfLoop);
    sort(backEdges.begin(), backEdges.end(), [&dfsInfo](const Edge& a, const Edge& b){
        return dfsInfo[a.dst].enterTime < dfsInfo[b.dst].enterTime;
    });
    // find out all the strongly connected components
    int curSccId = 0;
    for(auto& edge : backEdges){
        int u = edge.src, v = edge.dst;
        if(sccId[u] != -1)
            continue;
        if(sccId[v] == -1) {
            sccId[v] = curSccId;
            curSccId++;
        }
        set<int> nodesToVisit;
        nodesToVisit.insert(u);

        while(!nodesToVisit.empty()) {
            u = *nodesToVisit.begin();
            nodesToVisit.erase(nodesToVisit.begin());
            sccId[u] = sccId[v];

            while (u != v) {
                for(auto nxt : crossEdges[u]){
                    if(sccId[nxt] == -1)
                        nodesToVisit.insert(nxt);
                }
                u = dfsInfo[u].father;
                sccId[u] = sccId[v];
            }
        }
    }
    for(int i = 0; i< this->nodeNum; i++){
        if(sccId[i] == -1){
            sccId[i] = curSccId;
            curSccId++;
        }
    }
}

void binsim::Graph::calcDFSPostOrder(int node, vector<int>& record, vector<bool>& visited) const {
    visited[node] = true;
    for(auto &next: this->adjList[node]){
        if(!visited[next]){
            calcDFSPostOrder(next, record, visited);
        }
    }
    record.push_back(node);
}

int intersect(int b1, int b2, const vector<int>& idoms){
    int finger1 = b1, finger2 = b2;
    while(finger1 != finger2){
        while(finger1 > finger2){
            finger1 = idoms[finger1];
        }
        while(finger2 > finger1){
            finger2 = idoms[finger2];
        }
    }
    return finger1;
}


vector<int> binsim::Graph::calcDominatorTree() const {
    vector<int> dfsOrder;
    vector<bool> visited(this->nodeNum, false);
    calcDFSPostOrder(this->entryNode, dfsOrder, visited);
    reverse(dfsOrder.begin(), dfsOrder.end());

    vector<int> node2dfsOrder(this->nodeNum);
    for(int i = 0; i< this->nodeNum; i++){
        node2dfsOrder[dfsOrder[i]] = i;
    }
    vector<int> imm_dominators(this->nodeNum, -1);
    imm_dominators[0] = 0;
    // start iteration.
    bool changed = true;
    while(changed){
        changed = false;
        for(int i = 1; i< this->nodeNum; i++){
            int node = dfsOrder[i];
            auto prev_iter = this->adjListRev[node].begin();
            int pred = *prev_iter;
            int new_idom = node2dfsOrder[pred];
            while(imm_dominators[new_idom] == -1){
                prev_iter++;
                assert(prev_iter != this->adjListRev[node].end());
                pred = *prev_iter;
                new_idom = node2dfsOrder[pred];
            }
            for(++prev_iter; prev_iter != this->adjListRev[node].end(); prev_iter++){
                pred = *prev_iter;
                if(imm_dominators[node2dfsOrder[pred]] != -1){
                    new_idom = intersect(node2dfsOrder[pred], new_idom, imm_dominators);
                }
            }

            if(imm_dominators[i] != new_idom){
                imm_dominators[i] = new_idom;
                changed = true;
            }
        }
    }
    vector<int> domTree(this->nodeNum, -1);
    for(int i = 0; i< this->nodeNum; i++){
        domTree[dfsOrder[i]] = dfsOrder[imm_dominators[i]];
    }
    return domTree;
}

// Repeatedly apply T1 operation and T2 operation to the graph
// T1 operation: remove all self-loops
// T2 operation: remove all nodes with only one predecessor
pair<binsim::Graph, vector<set<int>>> binsim::Graph::reduce() const {
    // Create two new adjLists, as these adjLists use set, it is much more efficient.
    // All operations will be done on these two adjLists.
    vector<set<int>> fastAdjList(this->nodeNum), fastRevAdjList(this->nodeNum);
    // Whether the node is lived, lived means the node is not removed.
    vector<bool> lived(this->nodeNum, true);
    // This queue is used to store all nodes that will be processed in T2 operation.
    queue<int> nodes2deal;
    // Whether the node is in nodes2deal queue.
    vector<bool> dealing(this->nodeNum, false);
    // A saved the which nodes are merged into current node.
    UnionFindSet mergedNodes(this->nodeNum);

    for(int i = 0; i< this->nodeNum; i++){
        fastAdjList[i] = set<int>(this->adjList[i].begin(), this->adjList[i].end());
        fastRevAdjList[i] = set<int>(this->adjListRev[i].begin(), this->adjListRev[i].end());
        // remove all self-loops
        if(fastAdjList[i].count(i)){
            fastAdjList[i].erase(i);
            fastRevAdjList[i].erase(i);
        }
        // If the node has only one predecessor, it will be processed in T2 operation.
        // So we add it to the queue.
        if(fastRevAdjList[i].size() == 1 && i != 0){
            nodes2deal.push(i);
            dealing[i] = true;
        }
    }
    // Iteratively apply T2 operation until there is no node to be processed.
    while(!nodes2deal.empty()){
        int node = nodes2deal.front();
        nodes2deal.pop();
        dealing[node] = false;
        // This is a simple assert. I believe it will never fail.
        assert(fastRevAdjList[node].size() == 1);
        lived[node] = false;

        int prev = *fastRevAdjList[node].begin();
        // remove current node from its predecessor's adjList.
        fastAdjList[prev].erase(node);
        for(auto &next: fastAdjList[node]){
            // remove current node from its successor's revAdjList.
            fastRevAdjList[next].erase(node);
            // avoid adding self-loop
            if(next != prev){
                fastAdjList[prev].insert(next);
                fastRevAdjList[next].insert(prev);
            }
            // as T2 operation may modify #predecessors of the successor of current node, we have to check it.
            if(fastRevAdjList[next].size() == 1 && !dealing[next] && next != 0){
                nodes2deal.push(next);
                dealing[next] = true;
            }
        }
        mergedNodes.mergeInto(prev, node);
        fastAdjList[node].clear();
        fastRevAdjList[node].clear();
    }
    // Assign new node id to all lived nodes.
    vector<int> lived2index(this->nodeNum, -1);
    int counter = 0;
    for(int i = 0; i< this->nodeNum; i++){
        if(lived[i]){
            lived2index[i] = counter;
            counter++;
        }
    }
    vector<vector<int>> newAdjList(counter);
    for(int i = 0; i< this->nodeNum; i++){
        if(lived[i]){
            for(auto &next: fastAdjList[i]){
                newAdjList[lived2index[i]].push_back(lived2index[next]);
            }
        }
    }
    // Convert mergedNodes to a map, which maps the new node id to the set of old node ids.
    vector<set<int>> new2old(counter);
    for(int i = 0; i< this->nodeNum; i++){
        int newId = lived2index[mergedNodes.find(i)];
        new2old[newId].insert(i);
    }
    return {Graph(counter, newAdjList, lived2index[this->entryNode]), new2old};
}

void binsim::Graph::display() {
    cout << "Reducible: " << (this->isReducible()?"true":"false") << endl;
    cout << "Acyclic: " << (this->isAcyclic()?"true": "false") << endl;
    for(int i = 0; i< this->nodeNum; i++){
        cout << i << "->";
        if(!this->adjList[i].empty()){
            cout << "| ";
        }
        for(auto next: this->adjList[i]){
            cout << next << " | ";
        }
        cout << endl;
    }
}

inline void calculate_original2new(vector<int>&original2new, const vector<set<int>> & new2original){
    for(int i = 0; i< (int)new2original.size(); i++){
        for(auto node: new2original[i]){
            original2new[node] = i;
        }
    }
}

// Make current control flow graph reducible with controlled node splitting.
void binsim::Graph::toReducible() {
    // Apply T1 and T2 transformation repeatedly.
    auto reduced_pairs = this->reduce();
    auto reduced_graph = reduced_pairs.first;
    auto new2original = reduced_pairs.second;
    vector<int> original2new(this->nodeNum);
    calculate_original2new(original2new, new2original);

    // start converting.
    while(reduced_graph.nodeNum > 1){
        // select candidate nodes.
        auto selected_nodes = reduced_graph.selectNodes();
        assert(!selected_nodes.empty());
        int selected_node = selected_nodes[0];
        auto min_node = new2original[selected_node].size();
        for(int i = 1; i < (int)selected_nodes.size(); i++){
            int cur_node = selected_nodes[i];
            if(new2original[cur_node].size() < min_node){
                selected_node = cur_node;
                min_node = new2original[cur_node].size();
            }
        }
        // node splitting
        int old_reduced_graph_size = reduced_graph.nodeNum;
        int dup_num = reduced_graph.splitSingleNode(selected_node);
        int old_node_num = this->nodeNum;
        int number_of_nodes_to_dup = new2original[selected_node].size();
        this->splitMultiNode(new2original[selected_node], original2new);
        new2original.resize(reduced_graph.nodeNum);
        // update new2original;
        for(int j = 1; j< dup_num; j++){
            for(int k = 0; k < number_of_nodes_to_dup; k ++)
                new2original[old_reduced_graph_size + j - 1].insert(old_node_num + (j-1) * number_of_nodes_to_dup + k);
        }

        reduced_pairs = reduced_graph.reduce();
        reduced_graph = reduced_pairs.first;
        auto new2old = reduced_pairs.second;
        // update new2original
        vector<set<int>> _new2original(new2old.size());
        for(int i = 0; i< (int)new2old.size(); i++){
            for(auto node: new2old[i]){
                _new2original[i].insert(new2original[node].begin(), new2original[node].end());
            }
        }
        swap(new2original, _new2original);
        original2new = vector<int>(this->nodeNum);
        calculate_original2new(original2new, new2original);
    }
    this->modified = true;
}

void binsim::Graph::splitMultiNode(const set<int> &nodes, const vector<int> &original2new) {
    int group_id = original2new[*nodes.begin()];
    map<int, set<pair<int,int>>> edges;
    for(auto node: nodes){
        auto & prev_list = this->adjListRev[node];
        for(auto prev: prev_list){
            int new_id = original2new[prev];
            if(new_id != group_id){
                if(edges.find(new_id) == edges.end()){
                    edges[new_id] = set<pair<int,int>>();
                }
                edges[new_id].insert(make_pair(prev, node));
            }
        }
    }
    bool skip_first_edges_batch = true;
    for(auto edge_batch: edges){
        if(skip_first_edges_batch){
            skip_first_edges_batch = false;
            continue;
        }
        vector<int> copiedNodeId(this->nodeNum, -1);
        int copiedId  = this->nodeNum;
        auto itr = nodes.begin();
        for(; itr != nodes.end(); itr++, copiedId ++){
            copiedNodeId[*itr] = copiedId;
            this->nodeTags.emplace_back(this->nodeTags[*itr]);
            this->adjListRev.emplace_back();
            this->adjList.emplace_back();
        }
        this->nodeNum += (int) nodes.size();
        for(auto node: nodes){
            int copied_node_id = copiedNodeId[node];
            for(auto next: this->adjList[node]){
                // internal edges
                if(copiedNodeId[next] != -1){
                    int copied_next_id = copiedNodeId[next];
                    this->adjList[copied_node_id].push_back(copied_next_id);
                    this->adjListRev[copied_next_id].push_back(copied_node_id);
                }
                else{ // external edges
                    this->adjList[copied_node_id].push_back(next);
                    this->adjListRev[next].push_back(copied_node_id);
                }
            }
        }

        for(auto edge: edge_batch.second){
            int src = edge.first, dst = edge.second;
            int copied_dst = copiedNodeId[dst];
            std::replace(this->adjList[src].begin(), this->adjList[src].end(), dst, copied_dst);
            this->adjListRev[copied_dst].emplace_back(src);
        }
    }
    auto first_edge_batch = edges.begin()->second;
    for(auto &edge: first_edge_batch){
        int dst = edge.second;
        if(!this->adjListRev[dst].empty()){
            this->adjListRev[dst].clear();
        }
    }

    for(auto &edge: first_edge_batch){
        int dst = edge.second, src = edge.first;
        this->adjListRev[dst].emplace_back(src);
    }

}

int binsim::Graph::splitSingleNode(int node){
    int prev_num = (int)this->adjListRev[node].size();
    for(int i = 1; i < (int)this->adjListRev[node].size(); i++){
        int prev = this->adjListRev[node][i];
        int new_node = this->nodeNum;
        this->nodeNum ++ ;
        // copy predecessors.
        this->adjListRev.emplace_back(1,prev);
        std::replace(this->adjList[prev].begin(), this->adjList[prev].end(),node, new_node);
        // copy successors.
        this->adjList.emplace_back(this->adjList[node]);
        for(auto prev: this->adjList[node]){
            this->adjListRev[prev].emplace_back(new_node);
        }
    }
    this->adjListRev[node].resize(1);
    return prev_num;
}

vector<int> binsim::Graph::selectNodes() const {
    // Calculate the adjacent List of the dominator tree.
    auto tree = this->calcDominatorTree();
    vector<vector<int> > domTreeAdjList(this->nodeNum);
    for(int i = 0; i< this->nodeNum; i++){
        domTreeAdjList[tree[i]].push_back(i);
    }
    // calculate all strongly connected components
    auto componentIds = this->findStronglyConnectedComponents();
    vector<set<int>> components;
    for(int i = 0; i < (int)componentIds.size() ; i++){
        int id = componentIds[i];
        if(id >= (int)components.size()){
            components.resize(id+1);
        }
        components[id].insert(i);
    }
    queue<set<int>> components2deal;
    for(auto & comp: components){
        if(comp.size() == 1)
            continue;
        components2deal.push(comp);
    }
    vector<int> candidates;
    while(!components2deal.empty()){
        auto comp = components2deal.front();
        components2deal.pop();
        if(comp.size() == 1)
            continue;
        // Calculate the Shared External Dominators in current strongly connected component.
        vector<int> compSED;
        for(auto node: comp){
            int immDom = tree[node];
            // If immDom is not in the same component, immDom is an external dominator.
            if(comp.count(immDom) == 0)
                compSED.push_back(node);
        }
        // For each node in compSED,
        // 1. check whether it dominates any other strongly connected component that is a subset of current component;
        //     - if yes, it is in RC and is not a candidate. Meanwhile, we should put the strong components that it
        //       dominates into the queue, and deal with them later.
        //     - if not, it is in NN or CN, and should be a candidate.
        // 2. For those nodes that
        //     1. aren't in compSED
        //     2. and aren't in any strongly connected components dominated by any node in compSED.
        //    They must be not in any SED and should be discarded.
        //    Someone may ask what if some node is not in compSED but in a strongly connected components not dominated
        //    by any node in compSED? If you think it carefully, you will find it is impossible.
        for(auto sed_node: compSED){
            // use bfs to find out all nodes dominated by sed_node
            set<int> dominated;
            queue<int> nodes2visit;
            nodes2visit.push(sed_node);
            while(!nodes2visit.empty()){
                int node = nodes2visit.front();
                nodes2visit.pop();
                dominated.insert(node);
                for(auto &next: domTreeAdjList[node]){
                    if(dominated.find(next) == dominated.end()){
                        nodes2visit.push(next);
                    }
                }
            }
            dominated.erase(sed_node);
            comp.erase(sed_node);
            if(dominated.empty()) {
                candidates.push_back(sed_node);
                continue;
            }

            // find out all strongly connected components in dominated nodes
            auto sccId = this->findStronglyConnectedComponentsOnSubset(dominated);


            for(auto node: dominated){
                comp.erase(node);
            }

            if(sccId.empty()){
                candidates.push_back(sed_node);
                continue;
            }
            else{
                for(auto &scc: sccId){
                    if(scc.size() > 1){
                        components2deal.push(scc);
                    }
                }
            }
        }
        if(!comp.empty()){
            auto sub_comps = this->findStronglyConnectedComponentsOnSubset(comp);
            for(auto &sub_comp: sub_comps){
                if(sub_comp.size() > 1){
                    components2deal.push(sub_comp);
                }
            }
        }
    }
    return candidates;
}
