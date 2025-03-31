import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph
from rdkit import Chem
import numpy as np
import networkx as nx

class LigandGraphEncoder(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=2048, num_layer=3, k_hop=2):
        super().__init__()
        self.ingnn_dim = 78  # iNGNN-DTI atom features dimension
        self.emb_dim = emb_dim  # Initial embedding dimension
        self.hidden_dim = hidden_dim  # Final hidden dimension
        self.k_hop = k_hop  # Number of hops for subgraph extraction
        
        # Node feature projection
        self.node_projection = nn.Linear(self.ingnn_dim, emb_dim)
        
        # Edge encoder (binary presence)
        self.edge_encoder = nn.Linear(1, emb_dim)
        
        # Graph Attention layers for subgraph processing
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layer = num_layer
        
        for i in range(self.num_layer):
            in_channels = emb_dim if i == 0 else hidden_dim
            out_channels = hidden_dim // 4  # 2048 // 4 = 512 per head
            self.convs.append(GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=4,
                concat=True,  # Concatenate heads: 512 * 4 = 2048
                dropout=0.1,
                edge_dim=emb_dim
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Final projection for aggregation
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def atom_features(self, atom):
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                               'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                               'Pt', 'Hg', 'Pb', 'X']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Compute node features
        features = [self.atom_features(atom) / (sum(self.atom_features(atom)) + 1e-6) 
                    for atom in mol.GetAtoms()]
        features = torch.from_numpy(np.array(features)).float()  # Optimized conversion
        c_size = mol.GetNumAtoms()
        
        # Build edge indices with self-loops
        edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        g = nx.Graph(edges).to_directed()
        mol_adj = np.zeros((c_size, c_size))
        for e1, e2 in g.edges:
            mol_adj[e1, e2] = 1
        mol_adj += np.eye(c_size)  # Add self-loops
        index_row, index_col = np.where(mol_adj >= 0.5)
        edge_index = torch.from_numpy(np.array([index_row, index_col])).long()  # Optimized conversion
        
        # Edge features (binary presence)
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)
        
        # Extract subgraphs
        drug_node_indices = []
        drug_edge_indices = []
        drug_edge_attrs = []
        drug_indicators = []
        edge_index_start = 0
        for node_idx in range(c_size):
            sub_nodes, sub_edge_index, _, edge_mask = k_hop_subgraph(
                node_idx,
                self.k_hop,
                edge_index,
                relabel_nodes=True,
                num_nodes=c_size
            )
            drug_node_indices.append(sub_nodes)
            drug_edge_indices.append(sub_edge_index + edge_index_start)
            drug_edge_attrs.append(edge_attr[edge_mask])  # Subgraph-specific edge attributes
            drug_indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
            edge_index_start += sub_nodes.shape[0]
        
        # Construct Data object
        graph = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            subgraph_node_index=torch.cat(drug_node_indices),
            subgraph_edge_index=torch.cat(drug_edge_indices, dim=1),
            subgraph_edge_attr=torch.cat(drug_edge_attrs, dim=0),  # Store subgraph edge attributes
            subgraph_indicator_index=torch.cat(drug_indicators)
        )
        graph.__setitem__('c_size', torch.LongTensor([c_size]))
        return graph

    def forward(self, smiles_list):
        # Handle single SMILES or list of SMILES
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Convert SMILES to graphs
        graphs = [self.smiles_to_graph(smiles) for smiles in smiles_list]
        batch = Batch.from_data_list(graphs).to(next(self.parameters()).device)
        
        # Project node features
        x = self.node_projection(batch.x)  # [num_nodes, emb_dim]
        
        # Process subgraphs
        subgraph_x = x[batch.subgraph_node_index]  # [total_subgraph_nodes, emb_dim]
        subgraph_edge_index = batch.subgraph_edge_index
        subgraph_edge_attr = self.edge_encoder(batch.subgraph_edge_attr)  # [total_subgraph_edges, emb_dim]
        subgraph_indicator = batch.subgraph_indicator_index.long()
        
        # Graph convolution on subgraphs
        for i in range(self.num_layer):
            subgraph_x = self.convs[i](subgraph_x, subgraph_edge_index, subgraph_edge_attr)
            subgraph_x = self.norms[i](subgraph_x)
            subgraph_x = F.relu(subgraph_x)  # [total_subgraph_nodes, 2048]
        
        # Aggregate subgraph representations to original nodes
        num_nodes = batch.x.size(0)
        subgraph_agg = torch.zeros(num_nodes, self.hidden_dim, device=subgraph_x.device)
        subgraph_agg.index_add_(0, subgraph_indicator, subgraph_x)  # [num_nodes, 2048]
        counts = torch.bincount(subgraph_indicator, minlength=num_nodes).float().clamp(min=1)
        subgraph_agg = subgraph_agg / counts.unsqueeze(-1)  # Average per node
        
        # Final projection
        x = self.final_projection(subgraph_agg)  # [num_nodes, 2048]
        
        # Global pooling
        return global_mean_pool(x, batch.batch)  # [batch_size, 2048]

# Helper functions from iNGNN-DTI
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# Example usage
# if __name__ == "__main__":
#     encoder = LigandGraphEncoder(emb_dim=1024, hidden_dim=2048, num_layer=3, k_hop=2)
#     smiles_list = ["CCO", "CCN"]  # Batch of SMILES
#     output = encoder(smiles_list)
#     print(f"Output shape: {output.shape}")  # Should be [2, 2048]