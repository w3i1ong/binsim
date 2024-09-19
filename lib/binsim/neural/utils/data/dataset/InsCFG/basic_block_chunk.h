#ifndef BasicBlockChunk_H
#define BasicBlockChunk_H
#include <vector>
using namespace std;

namespace InsCFG {
    vector<int> solve(vector<int> &nums, int k);

    struct State {
        int padding;
        int last_chunk_end;

        State() {
            padding = 0;
            last_chunk_end = -1;
        }
    };
}

#endif
