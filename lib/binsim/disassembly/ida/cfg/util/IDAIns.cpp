#include "IDAIns.h"
#include <cstring>
#include <iostream>
using namespace std;

IDAIns::IDAIns(const map<std::string, int> &token2idx) {
    this->token2idx = token2idx;
}

void IDAIns::setToken2idx(const map<std::string, int> &token2idx) {
    this->token2idx = token2idx;
}

map<string,int> IDAIns::getToken2idx() const {
    return this->token2idx;
}

inline bool isHex(const string&str){
    if(str.back() != 'h')
        return false;
    for(size_t i = 0; i < str.size()-1;i++){
        if(!isdigit(str[i]) && !(str[i]>='A' && str[i]<='F'))
            return false;
    }
    return true;
}

inline bool isNumber(const string&str){
    return str.size() == 1 && str[0] >= '0' && str[0] <= '9';
}

inline bool isAddr(const string&str){
    return str[0] == '[' && str.back() == ']';
}

inline bool IDAIns::isRegister(const std::string &opnd_str) {
    return registers.find(opnd_str) != registers.end();
}

inline bool startswith(const string& str, const string& substr){
    if(str.size()<substr.size())
        return false;
    size_t i = 0;
    while(i<substr.size()){
        if(substr[i] != str[i])
            return false;
        i++;
    }
    return true;
}

void IDAIns::parseOpnd(string opnd_str, int opnd_idx) {
    // remove spaces
    size_t start_idx = 0, end_idx = opnd_str.size() - 1;
    while(start_idx < opnd_str.size() && opnd_str[start_idx] == ' ')
        start_idx++;
    while(end_idx >= start_idx && opnd_str[end_idx] == ' ')
        end_idx--;
    if(end_idx < start_idx) return;
    opnd_str = opnd_str.substr(start_idx, end_idx - start_idx + 1);

    // replace special tokens
    opnd_str = regex_replace(opnd_str, token_replacer, "");
    // replace '-' with '+'
    for(char & i : opnd_str){
        if(i == '-')
            i = '+';
    }

    if(isRegister(opnd_str)){
        opnds.emplace_back(opnd_str);
        return;
    }
    // check special prefix
    // "cs:", "ss:", "fs:", "ds:", "es:", "gs:"
    if(opnd_str.size()>3 && opnd_str[2] == ':') {
        for(auto & i : replace_strs1){
            if(startswith(opnd_str, i.first)){
                opnds.push_back(i.second);
                return;
            }
        }
    }


    // deal with jump instructions
    if(op[0] == 'j'){
        if(startswith(opnd_str, "loc_") || startswith(opnd_str, "sub_")){
            opnd_str[0] = 'h';
            opnd_str[1] = 'e';
            opnd_str[2] = 'x';
            opnds.emplace_back(opnd_str);
            return;
        }else if(startswith(opnd_str, "locret_")){
            opnds.emplace_back("hex_" + opnd_str.substr(7, opnd_str.size() - 7));
            return;
        }
        else{
            opnds.emplace_back("UNK_ADDR");
            return;
        }
    }

    if(opnd_str.size() > 4 && opnd_str[3] == '_') {
        // deal with other possible operands
        for (auto &i: replace_strs2) {
            if (startswith(opnd_str, i.first)) {
                opnds.push_back(i.second);
                return;
            }
        }
    }

    if(opnd_str.size() > 6 && opnd_str[5] == 't')
        if(startswith(opnd_str, "locret")){
            opnds.emplace_back("locretxxx");
            return;
        }
    //
    if(opnd_str[0] == '(' && opnd_str.back() == ')'){
        opnds.emplace_back("CONST");
        return;
    }
    if(op == "lea" && opnd_idx == 2){
        if(!isHex(opnd_str) && ! isAddr(opnd_str)){
            opnds.emplace_back("GLOBAL_VAR");
            return;
        }
    }

    if(op=="call" && opnd_idx == 1){
        if(opnd_str.size() > 3){
            opnds.emplace_back("callfunc_xxx");
            return;
        }
    }

    if(op == "extrn"){
        opnds.emplace_back("extrn_xxx");
        return;
    }

    if(isHex(opnd_str)){
        opnds.emplace_back("CONST");
        return;
    }

    if(isNumber(opnd_str)){
        opnds.emplace_back("CONST");
        return;
    }

    if(isAddr(opnd_str)){
        // split opnd_str with +
        vector<string> opnd_strs;
        start_idx = 1;
        while(start_idx < opnd_str.size() - 1){
            end_idx = start_idx;
            while(end_idx < opnd_str.size()-1 && opnd_str[end_idx] != '+')
                end_idx++;
            opnd_strs.emplace_back(opnd_str.substr(start_idx, end_idx - start_idx));
            start_idx = end_idx + 1;
        }
        for(auto & i : opnd_strs){
            if(isHex(i) || isNumber(i)){
                i = "CONST";
            }
            else if(startswith(i, "var_")){
                i = "var_xxx";
            }
            else if(startswith(i, "arg_")){
                i = "arg_xxx";
            }
            else if(!isRegister(i) && i.find('*') == string::npos){
                i = "CONST_VAR";
            }
        }
        int total_length = 0;
        for(auto & i : opnd_strs){
            total_length += (int)i.size() + 1;
        }
        total_length --;
        total_length += 2 + 1;
        char * ptr = new char[total_length];
        char * cur_ptr = ptr;
        *cur_ptr = '[';
        cur_ptr ++;
        for(auto & i : opnd_strs){
            strcpy(cur_ptr, i.c_str());
            cur_ptr += i.size();
            *cur_ptr = '+';
            cur_ptr ++;
        }
        cur_ptr--;
        *cur_ptr = ']';
        cur_ptr ++;
        *cur_ptr = '\0';
        opnds.emplace_back(ptr);
        delete[] ptr;
        return;
    }

    if(!isRegister(opnd_str) && opnd_str.size() > 4){
        opnds.emplace_back("CONST");
        return;
    }
    opnds.emplace_back(opnd_str);

}

void IDAIns::parseIns(std::string ins) {
    this->opnds.clear();
    // remove comments
    size_t start_idx, end_idx;
    if((end_idx = (int)ins.find(';')) != string::npos){
        ins = ins.substr(0, end_idx);
    }
    // parse operator
    if((end_idx = (int)ins.find(' ')) == string::npos){
        op = ins;
        return;
    }
    op = ins.substr(0, end_idx);
    ins = ins.substr(end_idx + 1, ins.size() - end_idx - 1);
    // parse operands
    int opnd_idx = 1;
    start_idx = 0;
    while(start_idx < ins.size()){
        end_idx = start_idx;
        while(end_idx < ins.size() && ins[end_idx] != ',')
            end_idx++;
        parseOpnd(ins.substr(start_idx, end_idx - start_idx), opnd_idx);
        opnd_idx ++;
        start_idx = end_idx + 1;
    }
}

vector<int> IDAIns::parseFunction(const vector<string> &ins_list, size_t max_length, const vector<pair<int, int>>&bb_length) {
    vector<int> token_ids;

    // add [CLS]
    token_ids.push_back(this->token2idx.at("[CLS]"));

    // initialize data structure to collect jump addresses
    map<int, int> addr2idx;
    int cur_bb_length = 0;
    int cur_bb_idx = 0;

    for(auto & i : ins_list){

        if(cur_bb_length == 0) {
            cur_bb_length = bb_length[cur_bb_idx].second;
            addr2idx[bb_length[cur_bb_idx].first] = token_ids.size() - 1;
            cur_bb_idx++;
        }

        this->parseIns(i);
        if(this->token2idx.find(this->op) == this->token2idx.end())
            token_ids.push_back(this->token2idx.at("[PAD]"));
        else
            token_ids.push_back(this->token2idx.at(this->op));

        for(int j = 0; j< 3 && j < this->opnds.size(); j++){
            if(startswith(this->opnds[j], "hex_")){
                int addr = stoi(this->opnds[j].substr(4, this->opnds[j].size() - 4), nullptr, 16);
                token_ids.push_back(-addr);
            }
            else {
                if(this->token2idx.find(this->opnds[j]) == this->token2idx.end())
                    token_ids.push_back(this->token2idx.at("[PAD]"));
                else
                    token_ids.push_back(this->token2idx.at(this->opnds[j]));
            }
        }
        cur_bb_length --;
        if(token_ids.size() >= max_length - 1)
            break;
    }

    // set unvisited jump addresses
    while(cur_bb_idx < bb_length.size()){
        addr2idx[bb_length[cur_bb_idx].first] = max_length + 1;
        cur_bb_idx ++;
    }

    // truncate token_ids
    while(token_ids.size() >= max_length){
        token_ids.pop_back();
    }

    token_ids.push_back(this->token2idx.at("[SEP]"));

    // add padding
    // we can conduct padding in dataloader, so we can save much memory.
    // int pad_idx = this->token2idx.at("[PAD]");
    // while (token_ids.size() < max_length){
    //    token_ids.push_back(pad_idx);
    // }

    for(auto &i: token_ids){
        if(i<0){
            i = -i;
            if(addr2idx.find(i) != addr2idx.end()){
                int idx = addr2idx[i];
                if(idx > max_length){
                    i = token2idx.at("JUMP_ADDR_EXCEEDED");
                }
                else{
                    i = token2idx.at("JUMP_ADDR_" + to_string(idx));
                }
            }
            else{
                i = token2idx.at("UNK_JUMP_ADDR");
            }
        }
    }

    return token_ids;
}

regex IDAIns::token_replacer = regex(R"((ptr\s|offset\s|xmmword\s|dword\s|qword\s|word\s|byte\s|short\s))");
set<string> IDAIns::registers = {
        "rax", "rbx", "rcx", "rdx", "esi", "edi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13", "r14",
        "r15"
};
vector<pair<string, string>> IDAIns::replace_strs1 = {
        {"cs:", "cs:xxx"},
        {"ss:", "ss:xxx"},
        {"fs:", "fs:xxx"},
        {"ds:", "ds:xxx"},
        {"es:", "es:xxx"},
        {"gs:", "gs:xxx"}
};
vector<pair<string, string>> IDAIns::replace_strs2 = {
        {"loc_", "loc_xxx"},
        {"off_", "off_xxx"},
        {"unk_", "unk_xxx"},
        {"sub_", "sub_xxx"},
        {"arg_", "arg_xxx"},
        {"def_", "def_xxx"},
        {"var_", "var_xxx"}
};

string IDAIns::getOperator() const {
    return this->op;
}

vector<string> IDAIns::getOperands() const {
    return this->opnds;
}

