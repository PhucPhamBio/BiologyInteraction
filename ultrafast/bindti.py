import torch.nn as nn
import torch.nn.functional as F
import torch
#from dgllife.model.gnn import GCN
#from ACmix import ACmix
#from Intention import BiIntention

class Intention(nn.Module):

    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super(Intention, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.device = device
        self.alpha = nn.Parameter(torch.rand(1))
        assert dim % num_heads == 0, 'dim must be divisible by num_heads!'

        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)

        self.softmax = nn.Softmax(dim=-2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, query=None):
        if query is None:
            query = x

        query = self.wq(query)
        key = self.wk(x)
        value = self.wv(x)

        b, n, c = x.shape
        key = key.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        key_t = key.clone().permute(0, 1, 3, 2)
        value = value.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        b, n, c = query.shape
        query = query.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        kk = key_t @ key
        kk = self.alpha * torch.eye(kk.shape[-1], device=self.device) + kk
        kk_inv = torch.inverse(kk)
        attn_map = (kk_inv @ key_t) @ value

        attn_map = self.softmax(attn_map)

        out = (query @ attn_map)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.out(out)

        return out

class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out

class IntentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Intention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x = self.norm_layer(x)
        q_t = q.permute(0, 2, 1)
        att = self.attn(x, q)
        att_map = self.softmax(att)
        out = self.beta * q_t @ att_map
        return out

class BiIntention(nn.Module):
    def __init__(self, embed_dim, num_head=8, layer=1, device='cuda'):
        super(BiIntention, self).__init__()

        self.layer = layer
        self.drug_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        #self attention
        # self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        # self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, drug, protein):
        #drug = self.attn_drug(drug)
        #protein = self.attn_protein(protein)

        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein)
            v_d = self.protein_intention[i](protein, drug)
            drug, protein = v_d, v_p

            v_d = torch.max(drug, dim=2)[0]  # [B, H, W] -> [B, H]
            v_p = torch.max(protein, dim=2)[0]  # [B, H, W] -> [B, H]

        f = torch.cat((v_d, v_p), dim=1)

        return f, v_d, v_p, None

class BINDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(BINDTI, self).__init__()
        # drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        # drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        # drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        # protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        # num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        # drug_padding = config["DRUG"]["PADDING"]
        # protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        # protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']

        # self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
        #                                    padding=drug_padding,
        #                                    hidden_feats=drug_hidden_feats)
        # self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):

        #v_d = self.drug_extractor(bg_d)#v_d.shape(64, 290, 128)
        #v_p = self.protein_extractor(v_p)#v_p.shape:(64, 1200, 128)

        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)#f:[64, 256]
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class BINDTI_Sprint(nn.Module) :
    def __init__(self, drug_dim = 2048, target_dim = 1280, num_head = 4, cross_layer = 1, device='cuda'):
        super(BINDTI_Sprint, self).__init__()
        self.drug_projection = nn.Sequential(nn.Linear(drug_dim, target_dim, bias=False), nn.SiLU())
        self.cross_intention = BiIntention(embed_dim=target_dim, 
                                           num_head=num_head, 
                                           layer=cross_layer, 
                                           device=device)

        self.mlp_classifier = MLPDecoder(2 * target_dim , target_dim, target_dim // 2, binary=1)

    def forward(self, v_d, v_p, mode="train"):
        # v_d : bs, max_seq_length, hidden_dim
        # v_p : bs, max_seq_length, hidden_dim
        v_d = self.drug_projection(v_d)
        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)#f:[64, 256]
        score = self.mlp_classifier(f)
        return v_d, v_p, f, score, att




"""
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats
"""

"""
class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])

        # self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        # self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)#64*128*1200

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))

        #v = self.bn3(F.relu(self.acmix3(v.unsqueeze(2))).squeeze())

        v = v.view(v.size(0), v.size(2), -1)
        return v
"""

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x