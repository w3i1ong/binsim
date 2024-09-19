#ifndef PARSE_INS_IDAINS_H
#define PARSE_INS_IDAINS_H
#include <string>
#include <regex>
#include <vector>
#include <set>
#include <map>
using namespace std;


class IDAIns {
private:
    string op;
    vector<string> opnds;
    map<string, int> token2idx;
    void parseOpnd(string opnd_str, int opnd_idx);
    static inline bool isRegister(const string& opnd_str);
public:
    IDAIns() = default;
    IDAIns(const map<string,int>& token2idx);
    void setToken2idx(const map<string, int> &token2idx);
    map<string, int> getToken2idx() const;
    static regex token_replacer;
    static set<string> registers;
    static vector<pair<string, string> > replace_strs1;
    static vector<pair<string, string>> replace_strs2;
    string getOperator() const;
    vector<string> getOperands() const;
    void parseIns(string ins);
    vector<int> parseFunction(const vector<string>& ins_list, size_t max_length, const vector<pair<int, int>>& addr2idx);
};


#endif //PARSE_INS_IDAINS_H
