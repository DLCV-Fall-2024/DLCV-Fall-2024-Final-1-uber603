import torch
from torch import nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, q_dim, kv_dim):
        super().__init__()
        self.q_shrink = nn.Linear(q_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=kv_dim, vdim=kv_dim, batch_first=True)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.q_expand = nn.Linear(embed_dim, q_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, query_embeds, key_value_embeds, key_padding_mask):
        # query_embeds: (batch, q_len, q_dim)
        # key_value_embeds: (batch, kv_len, kv_dim)
        # key_padding_mask: (batch, kv_len)

        # attention in latent space
        q = self.q_shrink(query_embeds)
        attn_output, _ = self.cross_attn(q, key_value_embeds, key_value_embeds, key_padding_mask=key_padding_mask)
        q = self.norm_1(q + attn_output)
        mlp_output = self.mlp(q)
        q = self.norm_2(q + mlp_output)
        q = self.q_expand(q)
        
        query_embeds = self.gate * q + query_embeds
        return query_embeds

class DecoderWithCrossAttention(nn.Module):
    def __init__(self, llama_decoder_layer, embed_dim_1, embed_dim_2, num_heads_1, num_heads_2, q_dim, kv_dim_1, kv_dim_2):
        super().__init__()
        self.llama_decoder_layer = llama_decoder_layer
        
        self.enable_cross_attn = False
        self.cross_attn_context_1 = None
        self.cross_attn_context_2 = None
        self.cross_attn_mask_1 = None
        self.cross_attn_mask_2 = None
        self.cross_attn_layer_1 = CrossAttentionLayer(embed_dim_1, num_heads_1, q_dim, kv_dim_1)
        self.cross_attn_layer_2 = CrossAttentionLayer(embed_dim_2, num_heads_2, q_dim, kv_dim_2)

    def forward(self, *args, **kwargs):
        x = self.llama_decoder_layer(*args, **kwargs)
        if self.enable_cross_attn:
            new_x0 = self.cross_attn_layer_1(x[0], self.cross_attn_context_1, self.cross_attn_mask_1)
            new_x0 = self.cross_attn_layer_2(new_x0, self.cross_attn_context_2, self.cross_attn_mask_2)
            x = (new_x0,) + x[1:]
        return x