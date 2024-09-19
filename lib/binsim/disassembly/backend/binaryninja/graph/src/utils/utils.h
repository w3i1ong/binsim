#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace std;

#define binsim_assert(condition, message, ...) \
    do{                                           \
        if(!(condition)){                          \
            fprintf(stderr, "Meet error in %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, message, ##__VA_ARGS__);                      \
            exit(1);                               \
        }                                   \
    } while(0)
