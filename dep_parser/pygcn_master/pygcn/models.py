import torch.nn as nn
import torch.nn.functional as F
from torch import cat, unsqueeze, tensor
import sys
sys.path.append('/home/qust-011/caption/dep_baseline/dep_parser/pygcn_master/')
from pygcn.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers):
        """
            nfeat = config.nhid
            nhid = config.nhid * n
        """
        super(GCN, self).__init__()

        layers = [GraphConvolution(nfeat, nhid)]    # length = nlayers
        for _ in range(nlayers-1):
            layers.append(GraphConvolution(nhid, nhid))

        self.bn_layer = nn.BatchNorm1d(nfeat)
        self.hid_layers = nn.ModuleList(layers)

        self.linear_hid = nn.Linear(nhid, nfeat)
        self.dropout = dropout

    # default forward(), with optional dropout setting
    def forward(self, embs, adjmat):
        assert len(embs.shape) == 3, 'Missing batch dimension.'

        hiddens = []
        for i in range(embs.shape[0]):      # each sample in a batch
            # x shape: (padded_len, nhid), includes <sos> and <eos>
            # adj shape: (token_len + 1, token_len + 1)
            # one global node in x and adj
            x, adj = embs[i], adjmat[i]
            
            # align x -> adj, truncate <eos> tokens
            # x = x[:adj.shape[0], :]   # shape: (adj.shape[0], nhid)
            
            hid = x[:adj.shape[0], :]
            for layer in self.hid_layers:
                hid = F.relu(layer(hid, adj))       
                # hid = F.dropout(hid, self.dropout, training=self.training)
            hid = self.linear_hid(hid)
            
            hiddens.append(hid[0])      # store updated first token
            # x = self.linear(F.dropout(hid1, self.dropout, training=self.training))
            
        return cat([unsqueeze(h,dim=0) for h in hiddens], dim=0)

    # try batch
    # def forward(self, embs, adjmat):
    #     assert len(embs.shape) == 3, 'Missing batch dimension.'

    #     hiddens = []
    #     hid = embs[:, :adjmat.shape[0], :]
    #     for layer in self.hid_layers:
    #         # hid = F.relu(self.bn_layer(layer(hid, adjmat)))     # use batch norm
    #         hid = F.relu(layer(hid, adjmat))
    #     hid = self.linear_hid(hid)
    #     return hid[:, 0, :]
        
        # for i in range(embs.shape[0]):      # each sample in a batch
        #     # x shape: (padded_len, nhid), includes <sos> and <eos>
        #     # adj shape: (token_len + 1, token_len + 1)
        #     # one global node in x and adj
        #     x, adj = embs[i], adjmat[i]
            
        #     # align x -> adj, truncate <eos> tokens
        #     # x = x[:adj.shape[0], :]   # shape: (adj.shape[0], nhid)
            
        #     hid = x[:adj.shape[0], :]
        #     for layer in self.hid_layers:
        #         hid = F.relu(layer(hid, adj))       
        #         # hid = F.dropout(hid, self.dropout, training=self.training)
        #     hid = self.linear_hid(hid)
            
        #     hiddens.append(hid[0])      # store updated first token
        #     # x = self.linear(F.dropout(hid1, self.dropout, training=self.training))
            
        # return cat([unsqueeze(h,dim=0) for h in hiddens], dim=0)

    # newly added version. not effective
    # def forward(self, embs, adjmat):
    #     assert len(embs.shape) == 3, 'Missing batch dimension.'

    #     hiddens = []
    #     for i in range(embs.shape[0]):  # each sample
    #         # x shape: (padded_len, nhid), includes <sos> and <eos>
    #         # adj shape: (token_len + 1, token_len + 1)
    #         # one global node in x and adj
    #         x, adj = embs[i], adjmat[i]
            
    #         # align x -> adj, truncate <eos> tokens
    #         # x = x[:adj.shape[0], :]   # shape: (adj.shape[0], nhid)
            
    #         hid = x[:adj.shape[0], :]
    #         for layer in self.hid_layers[:-1]:
    #             hid = F.relu(layer(hid, adj))
    #             hid = F.dropout(hid, self.dropout, training=self.training)
    #         hid = self.hid_layers[-1](hid, adj)
    #         hiddens.append(hid[0])      # store updated first token
            
    #     return cat([unsqueeze(h,dim=0) for h in hiddens], dim=0)
        # print('out',out.shape)
        # return tensor(hiddens)
