import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .backbone import build_backbone


def get_group_transformer_builder(args):
    """
    Dynamically import the build_group_transformer function based on the architecture choice
    """
    try:
        if args.group_transformer_arch == 'group_transformer':
            from .decoder import build_group_transformer
        else:
            raise ValueError(f"Unsupported group transformer architecture: {args.group_transformer_arch}")

        return build_group_transformer
    except ImportError as e:
        raise ImportError(f"Could not import build_group_transformer for {args.group_transformer_arch}: {e}")


class GADTR(nn.Module):
    def __init__(self, args):
        super(GADTR, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_class
        self.num_frame = args.num_frame
        self.num_boxes = args.num_boxes

        self.hidden_dim = args.hidden_dim
        self.backbone = build_backbone(args)

        # RoI Align
        self.crop_size_height = args.crop_size_height
        self.crop_size_width = args.crop_size_width

        # Group Transformer
        build_group_transformer = get_group_transformer_builder(args)
        self.group_transformer = build_group_transformer(args)
        self.num_group_tokens = args.num_group_tokens
        self.group_query_emb = nn.Embedding(self.num_group_tokens * self.num_frame, self.hidden_dim)

        # Classfication head
        self.class_emb = nn.Linear(self.hidden_dim, self.num_class + 1)
        self.group_emb = nn.Linear(self.hidden_dim, self.num_class + 1)

        # Membership prediction heads
        self.actor_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.group_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.relu = F.relu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'group_transformer' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, boxes, dummy_mask):
        """
        :param x: [B, T, 3, H, W]
        :param boxes: [B, T, N, 4]
        :param dummy_mask: [B, N]
        :return:
        """
        bs, t, _, h, w = x.shape
        n = boxes.shape[2]
        boxes = torch.reshape(boxes, (-1, 4))  # [b x t x n, 4]
        boxes_flat = boxes.clone().detach()

        # Create batch indices for roi_align
        # roi_align expects boxes in format [batch_idx, x1, y1, x2, y2]
        batch_indices = torch.cat([torch.full((n,), i, dtype=torch.float32, device=boxes.device)
                                   for i in range(bs * t)])
        batch_indices = batch_indices.reshape(-1, 1)  # [b x t x n, 1]

        features = self.backbone(x)

        src = features
        _, c, oh, ow = src.shape  # [b x t, d, oh, ow]
        src = torch.reshape(src, (bs, t, -1, oh, ow))  # [b, t, c, oh, ow]

        # Convert box format from [center_x, center_y, width, height] to [x1, y1, x2, y2]
        boxes_flat[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * ow  # x1
        boxes_flat[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * oh  # y1
        boxes_flat[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * ow  # x2
        boxes_flat[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * oh  # y2

        # ignore dummy boxes (padded boxes to match the number of actors)
        dummy_mask = dummy_mask.unsqueeze(1).repeat(1, t, 1).reshape(-1, n)
        actor_dummy_mask = (~dummy_mask.unsqueeze(2)).float() @ (~dummy_mask.unsqueeze(1)).float()
        dummy_diag = (dummy_mask.unsqueeze(2).float() @ dummy_mask.unsqueeze(1).float()).nonzero(as_tuple=True)
        actor_mask = ~(actor_dummy_mask.bool())
        actor_mask[dummy_diag] = False
        group_dummy_mask = dummy_mask

        # Combine batch indices with box coordinates to create the format expected by roi_align
        # Format: [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([batch_indices, boxes_flat], dim=1)  # [b x t x n, 5]
        rois.requires_grad = False

        actor_features = roi_align(features, rois, (self.crop_size_height, self.crop_size_width),
                                   spatial_scale=1.0, sampling_ratio=-1)
        actor_features = actor_features.view(bs * t * n, -1)  # now shape: (bs*t*n, C)
        actor_features = actor_features.reshape(bs, t, n, self.hidden_dim)

        hs, actor_att, feature_att = self.group_transformer(src, actor_mask, group_dummy_mask,
                                                            self.group_query_emb.weight, actor_features)
        # [1, bs * t, n + k, f'], [1, bs * t, k, n], [1, bs * t, n + k, oh x ow]   M: # group tokens, K: # boxes

        actor_hs = hs[0, :, :n]
        group_hs = hs[0, :, n:]

        actor_hs = actor_hs.reshape(bs, t, n, -1)

        # normalize
        inst_repr = F.normalize(actor_hs.reshape(bs, t, n, -1).mean(dim=1), p=2, dim=2)
        group_repr = F.normalize(group_hs.reshape(bs, t, self.num_group_tokens, -1).mean(dim=1), p=2, dim=2)

        # prediction heads
        outputs_class = self.class_emb(actor_hs)
        outputs_group_class = self.group_emb(group_hs)

        outputs_actor_emb = self.actor_match_emb(inst_repr)
        outputs_group_emb = self.group_match_emb(group_repr)

        membership_o = torch.bmm(outputs_group_emb, outputs_actor_emb.transpose(1, 2))
        membership = F.softmax(membership_o, dim=1)

        out = {
            "pred_actions": outputs_class.reshape(bs, t, self.num_boxes, self.num_class + 1).mean(dim=1),
            "pred_activities": outputs_group_class.reshape(bs, t, self.num_group_tokens, self.num_class + 1).mean(
                dim=1),
            "membership": membership.reshape(bs, self.num_group_tokens, self.num_boxes),
            "membership_o": membership_o.reshape(bs, self.num_group_tokens, self.num_boxes),
            "actor_embeddings": F.normalize(actor_hs.reshape(bs, t, n, -1).mean(dim=1), p=2, dim=2),
        }
        return out
