#include "UnionFindSet.h"

UnionFindSet::UnionFindSet(int n) {
    father.resize(n);
    for(int i = 0; i < n; i++){
        father[i] = i;
    }
}

int UnionFindSet::find(int x) {
    if(father[x] == x){
        return x;
    }
    return father[x] = find(father[x]);
}

void UnionFindSet::unionSet(int x, int y) {
    this->mergeInto(x, y);
}

void UnionFindSet::mergeInto(int dst, int src) {
    int fsrc = find(src);
    int fdst = find(dst);
    father[fsrc] = fdst;
}
