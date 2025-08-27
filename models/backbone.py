import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from transformers.models.dinov2.modeling_dinov2 import Dinov2Config, Dinov2Model


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        # Map DINOv2 model sizes to their corresponding model names
        # Class configuration
        self.num_frames = args.num_frame
        self.prompts_length = args.num_group_tokens
        self.use_group_prompts = args.use_group_prompts

        # Initialize DINOv2 model with pretrained weights
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.config = Dinov2Config.from_pretrained("facebook/dinov2-base")

        # Get model dimensions and structure
        self.num_channels = self.config.hidden_size
        self.num_layers = len(self.model.encoder.layer)

        # Create learnable global prompts for each layer
        if self.use_group_prompts:
            self.group_prompts = nn.Parameter(
                torch.zeros(self.num_layers, self.prompts_length, self.num_channels)
            )
            self._initialize_global_prompts()

        # Calculate the downsampled feature size based on DINOv2's patch size (typically 14x14)
        P = self.config.patch_size
        self.num_patches_width = args.image_width // P
        self.num_patches_height = args.image_height // P

    def _initialize_global_prompts(self):
        """Initialize global prompts using Xavier uniform initialization"""
        patch_size = (self.config.patch_size, self.config.patch_size)
        prompt_dim = self.num_channels

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # Xavier uniform initialization
        nn.init.uniform_(self.group_prompts.data, -val, val)

    def forward(self, x):
        # Input shape: (B, C, H, W)
        batch_size, _, H, W = x.shape

        # Get patch embeddings and CLS token from DINOv2's embedding layer
        hidden_states = self.model.embeddings(x)
        last_layer_group_prompts = None
        # Process through transformer layers with prompt insertion
        for i, layer_module in enumerate(self.model.encoder.layer):
            if self.use_group_prompts:
                # Extract global prompts for this layer and expand to batch size
                group_prompts = self.group_prompts[i].expand(batch_size, -1, -1)
                # Insert global prompts after the CLS token
                # [CLS] + [GLOBAL_PROMPTS] + [PATCHES]
                augmented_hidden_states = torch.cat(
                    (hidden_states[:, :1, :], group_prompts, hidden_states[:, 1:, :]),
                    dim=1
                )
                # Pass through the transformer layer
                augmented_hidden_states = layer_module(augmented_hidden_states)[0]

                # Remove the global prompts before passing to the next layer
                hidden_states = torch.cat(
                    (augmented_hidden_states[:, :1, :],
                     augmented_hidden_states[:, self.prompts_length + 1:, :]),
                    dim=1
                )
            else:
                # Standard forward pass without prompts
                hidden_states = layer_module(hidden_states)[0]

        # Apply final layer normalization
        hidden_states = self.model.layernorm(hidden_states)

        # Get the patch embeddings (excluding CLS token)
        patch_embeddings = hidden_states[:, 1:, :]  # Shape: (B*T, num_patches, hidden_size)

        # Reshape patches to spatial dimensions
        x = patch_embeddings.reshape(batch_size, self.num_patches_height, self.num_patches_width, -1)
        x = x.permute(0, 3, 1, 2)  # (B*T, C, H/patch_size, W/patch_size)

        return x


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        bs, t, _, h, w = x.shape
        x = x.reshape(bs * t, 3, h, w)

        backbone_output = self[0](x)
        features = backbone_output

        _, c, oh, ow = features.shape
        pos = self[1](features).to(x.dtype)

        return features, pos


class JoinerWithoutPosition(nn.Sequential):
    def __init__(self, backbone):
        super().__init__(backbone)

    def forward(self, x):
        bs, t, _, h, w = x.shape
        x = x.reshape(bs * t, 3, h, w)
        backbone_output = self[0](x)
        features = backbone_output

        return features


def build_backbone(args):
    if args.backbone == "finetune_backbone":
        print("Finetuning backbone model.")
    else:
        print(f"Using prompts intersection in frozen backbone: {args.use_group_prompts}")
    backbone = Backbone(args)

    model = JoinerWithoutPosition(backbone)
    model.num_channels = backbone.num_channels
    return model
