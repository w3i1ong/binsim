#include <vector>
#include <algorithm>
#include <cassert>
#include "basic_block_chunk.h"
using namespace std;

vector<int> InsCFG::solve(vector<int>& nums, int k){
    // dp[i][j] contains,
    // - the minimum padding
    // - the index of the last element in the j-th chunk
    // when the first i elements are divided into (j+1) chunks
    vector<vector<InsCFG::State> > dp(nums.size(), vector<InsCFG::State>(k));
    // obviously, dp[0][0].padding = 0 and dp[0][0].last_chunk_end = -1 (0-th chunk doesn't exist)
    // so, just skip it.
    // Now start from index 1
    for(int i = 1; i < nums.size(); i++){
        // the first (i+1) elements are divided into 1 chunk
        dp[i][0].padding = dp[i-1][0].padding + (nums[i] - nums[i-1]) * i;
        // otherwise, the first (i+1) elements are divided into more than 1 chunks
        // sometimes, i is so small, and elements can only be divided into (i+1) chunk
        for(int j = 1; j <= min(i,k-1); j++){
            // current element occupies a new chunk
            dp[i][j].padding = dp[i-1][j-1].padding;
            dp[i][j].last_chunk_end = i-1;
            // current element is appended to the last chunk
            if(dp[i-1][j].last_chunk_end != -1) {
                int cum_padding = 0;
                for (int l = i - 1; l >= dp[i - 1][j].last_chunk_end; l--) {
                    int padding = dp[l][j - 1].padding + cum_padding;
                    if (padding < dp[i][j].padding) {
                        dp[i][j].padding = padding;
                        dp[i][j].last_chunk_end = l;
                    }
                    cum_padding += nums[i] - nums[l];
                }
            }
        }
    }
    vector<int> results;
    int last = (int)nums.size()-1;
    for(int i = k-1;i>=0;i--){
        results.push_back(last);
        last = dp[last][i].last_chunk_end;
    }
    std::reverse(results.begin(), results.end());
    return results;
}
