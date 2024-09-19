try:
    from binsim.neural.nn.layer.dagnn.dagrnn_ops import (prepare_update_information_for_faster_forward as prepare_update_information,
                                                         transform_adj_list_for_fast_forward as transform_adj_list)
except ImportError:
    print("It seems that the CUDA extension is not compiled, if you want to use the faster version of DAGGRU, "
          "please reinstall binsim with CUDA support.")
class FastDAGGRUPropInfo:
    def __init__(self, index_map, node_batch_fwd, node_base_fwd,
                 edge_batch_fwd, edge_batch_index_fwd,
                 edge_batch_bck, edge_batch_index_bck,
                 in_degrees, out_degrees):
        self._index_map = index_map
        self._node_batch_fwd = node_batch_fwd
        self._node_base_fwd = node_base_fwd
        self._edge_batch_fwd = edge_batch_fwd
        self._edge_batch_index_fwd = edge_batch_index_fwd
        self._edge_batch_bck = edge_batch_bck
        self._edge_batch_index_bck = edge_batch_index_bck
        self._in_degrees = in_degrees.float()
        self._out_degrees = out_degrees.float()

    def to(self, device):
        self._index_map = self._index_map.to(device)
        self._node_batch_fwd = [batch.to(device) for batch in self._node_batch_fwd]
        self._edge_batch_fwd = [batch.to(device) for batch in self._edge_batch_fwd]
        self._edge_batch_index_fwd = [batch.to(device) for batch in self._edge_batch_index_fwd]
        self._edge_batch_bck = [batch.to(device) for batch in self._edge_batch_bck]
        self._edge_batch_index_bck = [batch.to(device) for batch in self._edge_batch_index_bck]
        self._in_degrees = self._in_degrees.to(device)
        self._out_degrees = self._out_degrees.to(device)
        return self

    @property
    def index_map(self):
        return self._index_map

    @property
    def node_batch_fwd(self):
        return self._node_batch_fwd

    @property
    def node_base_fwd(self):
        return self._node_base_fwd

    @property
    def node_batch_bck(self):
        return self._node_batch_fwd[::-1]

    @property
    def node_base_bck(self):
        return self._node_base_fwd[::-1]

    @property
    def edge_batch_fwd(self):
        return self._edge_batch_fwd

    @property
    def edge_batch_index_fwd(self):
        return self._edge_batch_index_fwd

    @property
    def edge_batch_bck(self):
        return self._edge_batch_bck

    @property
    def edge_batch_index_bck(self):
        return self._edge_batch_index_bck


    @property
    def in_degrees(self):
        return self._in_degrees

    @property
    def out_degrees(self):
        return self._out_degrees

    @property
    def all_information(self):
        return self.index_map, self.node_batch_fwd, self.node_base_fwd,\
            self.node_batch_bck, self.node_base_bck, self.edge_batch_fwd, self.edge_batch_index_fwd, \
            self.edge_batch_bck, self.edge_batch_index_bck, self.in_degrees, self.out_degrees

    def check(self, adj_list):
        index_map = self.index_map.cpu().numpy().tolist()
        old_index = {idx: i for i, idx in enumerate(index_map)}

        new_adj_list = [[] for _ in range(len(adj_list))]
        for edge_batch, edge_batch_index in zip(self.edge_batch_fwd, self.edge_batch_index_fwd):
            edge_batch = edge_batch.cpu().numpy().tolist()
            for src, dst in edge_batch:
                new_adj_list[src].append(dst)

        recovered_adj = [[] for _ in range(len(adj_list))]
        for node_id, next_list in enumerate(new_adj_list):
            recovered_adj[old_index[node_id]] = [old_index[i] for i in next_list]
            recovered_adj[old_index[node_id]].sort()
            adj_list[old_index[node_id]].sort()
        assert adj_list == recovered_adj

def prepare_update_information_for_faster_forward(adj_list):
    adj_list, index_map = transform_adj_list(adj_list)
    (node_batch_fwd, edge_batch_fwd, edge_batch_index_fwd, edge_batch_bck, edge_batch_index_bck,
     in_degrees, out_degrees) = \
        prepare_update_information(adj_list)
    node_batch_fwd = [batch for batch in node_batch_fwd]
    node_base_fwd, cur_base = [], 0

    for node_batch in node_batch_fwd:
        node_base_fwd.append(cur_base)
        cur_base += node_batch.shape[0]

    edge_batch_fwd = [batch for batch in edge_batch_fwd]
    edge_batch_index_fwd = [batch_index for batch_index in edge_batch_index_fwd]

    edge_batch_bck = [batch for batch in edge_batch_bck][::-1]
    edge_batch_index_bck = [batch_index for batch_index in edge_batch_index_bck][::-1]


    return FastDAGGRUPropInfo(index_map=index_map,node_batch_fwd=node_batch_fwd, node_base_fwd=node_base_fwd,
                              edge_batch_fwd=edge_batch_fwd, edge_batch_index_fwd=edge_batch_index_fwd,
                              edge_batch_bck=edge_batch_bck, edge_batch_index_bck=edge_batch_index_bck,
                              in_degrees=in_degrees, out_degrees=out_degrees)
