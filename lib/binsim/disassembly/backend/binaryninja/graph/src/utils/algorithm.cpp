#include <cstdio>
#include <algorithm>
#include <vector>
#include <pybind11/pybind11.h>
#include "./utils.h"
using namespace std;


vector<pair<int, int>> split_chunks(const vector<pair<int,int>>& groups, int chunks){
    pybind11::gil_scoped_release release;
    binsim_assert(chunks > 1, "Chunks should be greater than 0, but got %d\n", chunks);
    // ensure the groups are sorted by the first element(length) in descending order
    for(size_t i = 1; i < groups.size(); i++){
        binsim_assert(groups[i].first < groups[i - 1].first,
                      "The groups should be sorted by the first element in descending order, "
                      "but got %zu-th element %d is less than %zu-th element %d\n",
                      i - 1, groups[i - 1].first, i, groups[i].first);
    }

    // initialize the dp table
    // dp[i][j] = min padding needed when splitting the first i groups into j chunks
    vector<vector<int> > dp(groups.size(), vector<int>(chunks + 1, 0));
    vector<vector<int>> choice(groups.size(), vector<int>(chunks + 1, -1));
    // initialization
    // dp[x][1] = padding(1, x)
    int max_length = groups[0].first;
    int cum_padding = 0;
    for(size_t i = 0; i < groups.size(); i++){
        cum_padding += (max_length - groups[i].first) * groups[i].second;
        dp[i][1] = cum_padding;
        choice[i][1] = -1;
    }
    // start dynamic programming
    // dp[i][j] = min_{j < k < i} (dp[k][j - 1] + padding(k + 1, i))
    for(int i = 0; i < (int)groups.size(); i++){
        int group_length = groups[i].first;
        for(int j = 2; j <= chunks; j++){
            if(j >= i+1){
                // number of chunks is greater than the number of groups
                // we can place each group in a chunk, and no padding is needed
                dp[i][j] = 0;
                choice[i][j] = i-1;
                continue;
            }
            // i-th group is split into the last chunk
            dp[i][j] = dp[i][j - 1];
            choice[i][j] = choice[i][j-1];
            cum_padding = 0;
            int cum_num = 0;
            // the i to k-th groups in the last chunk
            for(int k = i; k >= j; k--){
                cum_padding += (groups[k].first - group_length) * cum_num;
                cum_num += groups[k].second;
                if(dp[k-1][j - 1] + cum_padding < dp[i][j]) {
                    dp[i][j] = dp[k-1][j - 1] + cum_padding;
                    choice[i][j] = k - 1;
                }
            }
        }
    }
    // find the optimal split
    vector<pair<int, int>> results;
    for(int i = (int)groups.size() - 1, j = chunks; i!= -1 ;){
        int last = choice[i][j];
        results.emplace_back(last + 1, i + 1);
        i = last;
        j --;
    }
    return {results.rbegin(), results.rend()};
}
