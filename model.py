import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import parameter as my_data
import numpy as np
import copy

input_len = my_data.input_len
output_len = my_data.output_len
embed_size = my_data.embed_size
batch_size = my_data.batch_size
device = my_data.device
st_windows = my_data.st_windows
patch_windows = my_data.patch_windows


# batch_size=1
def windows(input,change,windows,temporal_length):
    return rearrange(input,change,p1 = windows,p2 = windows,h = 100//windows,t = temporal_length,b = batch_size)





class STPTransformer(nn.Module):
    def __init__(self,  num_layers,embed_size,nhead,sequence_len,st_window,p_window, dropout):
        super(STPTransformer, self).__init__()
        self.space_encoder_layer = EncoderLayer(embed_size[0], nhead[0], 2048, dropout,attention_type = "space")
        self.space_transformer = Space_Transformer(self.space_encoder_layer,embed_size[0],num_layers[0],sequence_len,st_window,attention_type = "space")

        self.temporal_encoder_layer = EncoderLayer(embed_size[1], nhead[1], 2048, dropout,attention_type = "space")
        self.temporal_transformer = Temporal_Transformer(self.temporal_encoder_layer,embed_size[1],num_layers[1],sequence_len,st_window,attention_type = "space")

        self.patch_encoder_layer = EncoderLayer(embed_size[2], nhead[2], 2048, dropout,attention_type = "space")
        self.patch_transformer = Patch_transformer(self.patch_encoder_layer,embed_size[2],num_layers[2],sequence_len,p_window,attention_type = "space")


        self.conv = nn.Conv2d(input_len,output_len,kernel_size=3,padding=1)

        self.corss_attn = CorssHeadedAttention(embed_size[1], nhead[1], CorssAttention, dropout,attention_type = "space")
        self.sigmod = nn.Sigmoid()

        self.st_window = st_window
    def forward(self, src):

        src = src.reshape(batch_size,input_len,100,100)
        temporal_out = self.temporal_transformer(src)
        space_out = self.space_transformer(src)
        cross_out = self.corss_attn(space_out,temporal_out,temporal_out)
        cross_out = windows(cross_out, 'b (h w) (p1 p2) t->b t (h p1) (w p2)', self.st_window, input_len)

        patch_out = self.patch_transformer(cross_out,src)
        # patch_out = cross_out+patch_out
        output = self.conv(patch_out)
        output = self.sigmod(output)

        return output.reshape(batch_size,output_len,-1)





class Space_Transformer(nn.Module):
    def __init__(self,encoder_layer,embed_size,num_layers,sequence_len,window,attention_type = "space"):
        super(Space_Transformer, self).__init__()
        self.space_windows = window
        self.sequence_len = sequence_len
        self.space_encoder = Encoder(encoder_layer,sequence_len, num_layers,embed_size,attention_type)
        self.space_reverse = nn.Linear(embed_size,sequence_len)


    def forward(self,src):
        space_input = windows(src, 'b t (h p1) (w p2)->b (h w) (p1 p2) t', self.space_windows, self.sequence_len)
        encoder_out = self.space_encoder(space_input, None)
        return encoder_out

class Temporal_Transformer(nn.Module):
    def __init__(self,encoder_layer, embed_size, num_layers,sequence_len,window,attention_type):
        super(Temporal_Transformer, self).__init__()
        self.temporal_windows = window
        self.sequence_len = sequence_len
        self.encoder = Encoder(encoder_layer,window**2, num_layers,embed_size,attention_type)


    def forward(self, src, src_mask=None):
        src = windows(src, 'b t (h p1) (w p2)->b (h w)  t  (p1 p2)', self.temporal_windows, self.sequence_len)
        temporal_out = self.encoder(src, src_mask)
        return temporal_out

class Patch_transformer(nn.Module):
    def __init__(self,encoder_layer,embed_size,num_layers,sequence_len,window,attention_type = "space"):
        super(Patch_transformer, self).__init__()
        self.patch_windows = window
        self.sequence_len = sequence_len
        self.patch_encoder = Encoder(encoder_layer,sequence_len,num_layers,embed_size,attention_type)
        self.patch_reverse = nn.Linear(embed_size,sequence_len)

    def forward(self,input,src):
        space_patch_input = input+src
        space_patch_input = windows(space_patch_input,'b t (h p1) (w p2)->b (p1 p2) (h w) t', self.patch_windows, self.sequence_len)
        space_patch_out = self.patch_encoder(space_patch_input,None)
        space_patch_out = self.patch_reverse(space_patch_out)
        space_patch_out = windows(space_patch_out,'b (p1 p2) (h w) t->b t (h p1) (w p2)', self.patch_windows, self.sequence_len)
        space_patch_out = space_patch_out+input+src

        return space_patch_out

class input_embedding(nn.Module):
    def __init__(self, input_dim, d_model, max_len,attention_type):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe_time = pe[None, None].to(device)

        self.linear = nn.Linear(input_dim, d_model)

        # self.conv2d = nn.Conv2d(in_channels=input_dim,out_channels=d_model,kernel_size=1,padding=0)
        self.norm = nn.LayerNorm(d_model)

        self.register_buffer('pe', pe)

        self.attention_type = attention_type

    def forward(self, x):
        assert len(x.size()) == 4
        # x = x.permute(0,3,2,1)
        # x = self.conv2d(x).permute(0,3,2,1)
        x = self.linear(x)

        if self.attention_type =="space":
            pe = self.pe_time[:, :, :x.size(2)].to(device)
            x = x+pe
            # x = x + self.pe_time[:, :, :x.size(2)]  # (N, S, T, D)

        elif self.attention_type == "temporal":
            x = x
        return self.norm(x)



class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class Encoder(nn.Module):
    def __init__(self, encoder_layer, feature_size,num_layers,embed_size,attention_type):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.en_pe = input_embedding(input_dim = feature_size,d_model = embed_size,max_len = 10000,attention_type = attention_type)

    def forward(self, x, mask=None):
        x = self.en_pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout,attention_type):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout,attention_type)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )


    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.time_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)



class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout,attention_type):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        if attention_type == "temporal":
            self.attention = Attention(n_heads =nheads ,hid_dim=d_model , dropout=dropout)
        elif attention_type=="space":
            self.attention = attn
    def forward(self, query, key, value, mask=None):

        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)

        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
             for l, x in zip(self.linears, (query, key, value))]

        x = self.attention(query, key, value, mask=mask)

        x = x.permute(0, 2, 3, 1, 4).contiguous() \
             .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        return self.linears[-1](x)

def TimeAttention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, T)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        scores = scores.masked_fill(mask[None, None], float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    # if p_attn.size(3)==10000:
    #
    #     attention_display = np.array(p_attn[0,0,0,:,:].to("cpu"))
    #     attention_display = pd.DataFrame(attention_display).to_csv("attenion_display.csv",header=False,index= False,)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)  # (N, h, S, T, D)

class CorssHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout,attention_type):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout)
        if attention_type == "temporal":
            self.attention = Attention(n_heads =nheads ,hid_dim=d_model , dropout=dropout)
        elif attention_type=="space":
            self.attention = attn
    def forward(self, query, key, value, mask=None):



        query,key = [l(x) for l,x in zip(self.linears,(query,key))]

        # x = self.attention(query, key, value, mask=mask)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, T)

        return scores
def CorssAttention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, T)

    return scores


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    def __init__(self, n_heads,hid_dim , dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_position

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):


        len_k = key.shape[3]
        len_q = query.shape[3]
        len_v = value.shape[3]

        r_q1 = query
        r_k1 = key
        attn1 = torch.matmul(r_q1, r_k1.transpose(-2, -1))

        r_k2 = self.relative_position_k(len_q, len_k)
        r_k2 = r_k2[None,None,:]
        r_q2 = query.permute(0,1,3,2,4)

        attn2 = torch.matmul(r_q2, r_k2.transpose(-1, -2)).permute(0,1,3,2,4)

        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        r_v1 = value
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        r_v2 = r_v2[None,None,:]
        weight2 = attn.permute(0,1,3,2,4)
        weight2 = torch.matmul(weight2, r_v2).permute(0,1,3,2,4)
        x = weight1 + weight2
        x = self.fc_o(x)
        return x



# class RelativePosition(nn.Module):
#
#     def __init__(self, num_units, max_relative_position):
#         super().__init__()
#         self.num_units = num_units
#         self.max_relative_position = max_relative_position
#         self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units)).to(device)
#         nn.init.xavier_uniform_(self.embeddings_table)
#
#     def forward(self, length_q, length_k):
#
#         range_vec_q = torch.arange(length_q)
#         range_vec_k = torch.arange(length_k)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         xx = torch.where(distance_mat > length_k - self.max_relative_position, distance_mat - (length_k+self.max_relative_position), 0)
#         yy = torch.where(distance_mat < -(length_k - self.max_relative_position), distance_mat + (length_k+self.max_relative_position), 0)
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#
#         final_mat = final_mat+xx+yy
#         final_mat = torch.LongTensor(final_mat)
#         embeddings = self.embeddings_table[final_mat].to(device)
#
#         return embeddings

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings