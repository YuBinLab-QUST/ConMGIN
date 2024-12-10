import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
import torch

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GIN, self).__init__()
        self.gc1 = GINConv(nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        ))
        self.gc2 = GINConv(nn.Sequential(
            nn.Linear(nhid, out),
            nn.ReLU(),
            nn.Linear(out, out)
        ))
        self.dropout = dropout

    def forward(self, x, edge_index):
        edge_index = edge_index.coalesce()
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.var = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.variance = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        var = self.variance(self.var(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, var, mean]


class SpatialAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(SpatialAttention, self).__init__()

        # Linear projection for computing attention scores
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # Compute attention scores for each region
        w = self.project(z)  # Shape: [batch_size, N, 1]

        # Apply softmax to get attention weights
        beta = torch.softmax(w, dim=1)  # Softmax along the spatial dimension (N)

        # Compute the weighted sum of the input features based on attention weights
        output = (beta * z).sum(dim=1)  # Shape: [batch_size, in_size]

        return output, beta


class ConMGIN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(ConMGIN, self).__init__()
        self.MGIN = GIN(nfeat, nhid1, nhid2, dropout)
        self.Bayesian = decoder(nfeat, nhid1, nhid2)
        self.att = SpatialAttention(nhid2)
        self.MLP = nn.Linear(nhid2, nhid2)
        self.dropout = dropout

    def forward(self, x, sadj, fadj):
        emb1 = self.MGIN(x, sadj)  # Space-adjacent embedding
        emb2 = self.MGIN(x, fadj)  # Feature-adjacent embedding
        com = (self.MGIN(x, sadj) + self.MGIN(x, fadj)) / 2  # Combined embedding

        # Stack embeddings and apply attention
        emb = torch.stack([emb1, com, emb2], dim=1)
        emb, _ = self.att(emb)
        emb = self.MLP(emb)

        # Decode to Bayesian outputs
        pi, var, mean = self.Bayesian(emb)
        return emb1, emb2, emb, pi, var, mean


