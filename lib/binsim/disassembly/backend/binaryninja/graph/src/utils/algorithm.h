#ifndef DATAUTILS_ALGORITHM_H
#define DATAUTILS_ALGORITHM_H
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "./utils.h"
using namespace std;


vector<pair<int, int>> split_chunks(const vector<pair<int,int>>& groups, int chunks);


template<class T>
void split_chunks(const vector<vector<T>>& data, int chunks, vector<vector<vector<T>>>&result,
                  vector<vector<int>>& lengths, vector<int>& index){
    lengths.clear(), index.clear();

    // extract (index, length) pairs for each group
    vector<pair<int, int>> meta_info;
    meta_info.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        meta_info.emplace_back(i, (int)data[i].size());
    }
    // sort the groups by the length in descending order
    sort(meta_info.begin(), meta_info.end(), [](const pair<int, int>& a, const pair<int, int>& b){
        return a.second > b.second;
    });
    // merge the sequences with the same length into a group
    // each group is described by a pair of (length, number of sequences with the same length)
    vector<pair<int, int>> groups;
    int last_length = meta_info[0].second;
    size_t last_index = 0;
    for(size_t i = 1; i < meta_info.size(); i++){
        auto& info = meta_info[i];
        if(info.second != last_length){
            groups.emplace_back(last_length, i - last_index);
            last_length = info.second;
            last_index = i;
        }
    }
    groups.emplace_back(last_length, meta_info.size() - last_index);
    // now we can use dynamic programming to split the groups into chunks
    auto split_result = split_chunks(groups, chunks);

    size_t meta_index = 0;
    for(auto& item: split_result){
        int group_start = item.first, group_end = item.second;
        int min_length = group_end < (int)groups.size()? groups[group_end].first: groups.back().first - 1;
        int max_length = (int)groups[group_start].first;
        size_t meta_start = meta_index;
        binsim_assert(max_length == meta_info[meta_index].second,
                      "The max length of the group should be %d, but got %d\n", max_length, meta_info[meta_index].second);
        while(meta_index < meta_info.size() && meta_info[meta_index].second > min_length){
            meta_index ++;
        }
        size_t meta_end = meta_index;
        auto cur_chunk = vector<vector<T>>();
        auto cur_length = vector<int>();
        cur_length.reserve(meta_end - meta_start);
        for(size_t i = meta_start; i < meta_end; i++){
            cur_length.push_back(meta_info[i].second);
            index.push_back(meta_info[i].first);
            cur_chunk.push_back(data[meta_info[i].first]);
        }
        lengths.emplace_back(cur_length);
        result.push_back(cur_chunk);
    }
}
#endif