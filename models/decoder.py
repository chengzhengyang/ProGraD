from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self, d_model=768, nhead=4, num_actor=14, dim_feedforward=1536, dropout=0.1,
                 activation="relu", use_mask=False):

        super().__init__()
        self.decoder = TransformerDecoder(d_model, nhead, num_actor, dim_feedforward, dropout, activation, use_mask)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, actor_mask, group_dummy_mask, group_embed, actor_embed):
        bs, t, c, h, w = src.shape
        # print('src.shape:', src.shape)
        _, _, n, _ = actor_embed.shape
        src = src.reshape(bs * t, c, h, w).flatten(2).permute(2, 0, 1)  # [h x w, bs x t, c]

        memory = src

        group_embed = group_embed.reshape(-1, t, c)
        group_embed = group_embed.unsqueeze(1).repeat(1, bs, 1, 1).reshape(-1, bs * t, c)
        actor_embed = actor_embed.reshape(bs * t, -1, c).permute(1, 0, 2)  # [n, bs x t, c]

        query_embed = torch.cat([actor_embed, group_embed], dim=0)  # [n + k, bs x t, c]

        if actor_mask is not None:
            actor_mask = actor_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(-1, n, n)
        hs, actor_att, feature_att = self.decoder(query_embed, memory, tgt_key_padding_mask=group_dummy_mask)

        return hs.transpose(1, 2), actor_att, feature_att


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_actor, dim_feedforward=2048, dropout=0.1, activation="relu", use_mask=False):
        super().__init__()

        # All the components from TransformerDecoderLayer integrated here
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm_attn1_actor = nn.LayerNorm(d_model)
        self.norm_attn1_group = nn.LayerNorm(d_model)
        self.norm_attn2_src = nn.LayerNorm(d_model)
        self.norm_attn2_tgt = nn.LayerNorm(d_model)
        self.norm_attn2 = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.use_mask = use_mask
        self.num_actor = num_actor

    def decoder_layer_forward(self, tgt, memory, use_mask=False,
                            tgt_key_padding_mask: Optional[Tensor] = None):
        """
        Integrated decoder layer functionality
        """
        # actor-group attention
        tgt_actor = tgt[:self.num_actor, :, :]
        tgt_group = tgt[self.num_actor:, :, :]
        tgt_actor = self.norm_attn1_actor(tgt_actor)
        tgt_group = self.norm_attn1_group(tgt_group)

        if use_mask:
            tgt2_group, group_att = self.multihead_attn1(query=tgt_group, key=tgt_actor, value=tgt_actor,
                                                         key_padding_mask=tgt_key_padding_mask)
        else:
            tgt2_group, group_att = self.multihead_attn1(query=tgt_group, key=tgt_actor, value=tgt_actor)

        tgt_group = tgt_group + self.dropout_attn1(tgt2_group)

        tgt2_group = self.norm_ffn1(tgt_group)
        tgt2_group = self.linear4(self.dropout_fc1(self.activation(self.linear3(tgt2_group))))
        tgt_group = tgt_group + tgt2_group

        tgt2 = torch.cat([tgt_actor, tgt_group], dim=0)
        tgt = self.norm_attn2_tgt(tgt2)
        memory = self.norm_attn2_src(memory)

        # actor-feature, group-feature cross-attention
        tgt2, feature_att = self.multihead_attn2(query=tgt,
                                                 key=memory, value=memory)
        tgt = tgt + self.dropout_attn2(tgt2)
        tgt = self.norm_attn2(tgt)

        tgt2 = self.norm_ffn(tgt)
        tgt2 = self.linear2(self.dropout_fc(self.activation(self.linear1(tgt2))))
        tgt = tgt + tgt2

        return tgt, group_att, feature_att

    def forward(self, query, memory, tgt_key_padding_mask: Optional[Tensor] = None):

        # Since num_layers is 1, run decoder layer once
        output, actor_att, feature_att = self.decoder_layer_forward(query, memory, use_mask=self.use_mask,
                                                                    tgt_key_padding_mask=tgt_key_padding_mask)

        if actor_att is not None:
            actor_att = actor_att.unsqueeze(0)
        if feature_att is not None:
            feature_att = feature_att.unsqueeze(0)

        return output.unsqueeze(0), actor_att, feature_att


def build_group_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.gar_nheads,
        num_actor=args.num_boxes,
        use_mask=args.use_mask,
        dim_feedforward=args.gar_ffn_dim,
        dropout=args.drop_rate)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")