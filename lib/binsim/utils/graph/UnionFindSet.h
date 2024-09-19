#ifndef UNIONFINDSET_H
#define UNIONFINDSET_H
#include <vector>
using namespace std;

class UnionFindSet {
public:
    explicit UnionFindSet(int n);
    int find(int x);
    void unionSet(int x, int y);
    void mergeInto(int dst, int src);
private:
    vector<int> father;

};


#endif //UNIONFINDSET_H
