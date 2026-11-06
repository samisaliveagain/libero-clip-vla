"""Stage 4 learned fusion policy over frozen CLIP token features."""

from __future__ import annotations

import torch
from torch import nn


class LiberoActionPolicy(nn.Module):
    """Predict one normalized OSC_POSE + gripper command per observation."""

    def __init__(self) -> None:
        super().__init__()
        width = 256
        self.visual_projector = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.text_projector = nn.Sequential(nn.Linear(512, width), nn.LayerNorm(width))
        self.proprio_encoder = nn.Sequential(
            nn.Linear(9, 128), nn.GELU(), nn.Linear(128, width)
        )
        # Separate trainable identities for image, text, state, and action query.
        self.modality = nn.Parameter(torch.zeros(4, width))
        self.action_query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.modality, std=0.02)
        nn.init.normal_(self.action_query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=8, dim_feedforward=1024,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=4)
        self.action_head = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 7), nn.Tanh()
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        batch = proprio.shape[0]
        if visual.shape != (batch, 196, 768):
            raise ValueError(f"Expected visual [B,196,768], got {tuple(visual.shape)}")
        if text.shape != (batch, 77, 512) or text_mask.shape != (batch, 77):
            raise ValueError("Expected text [B,77,512] and mask [B,77]")
        if proprio.shape != (batch, 9):
            raise ValueError(f"Expected proprio [B,9], got {tuple(proprio.shape)}")
        if (text_mask.sum(dim=1) == 0).any():
            raise ValueError("Every instruction needs at least one valid token")

        vision_tokens = self.visual_projector(visual) + self.modality[0]
        text_tokens = self.text_projector(text) + self.modality[1]
        state_token = self.proprio_encoder(proprio).unsqueeze(1) + self.modality[2]
        query_token = self.action_query.expand(batch, -1, -1) + self.modality[3]
        tokens = torch.cat((vision_tokens, text_tokens, state_token, query_token), dim=1)
        padding = torch.cat((
            torch.zeros((batch, 196), dtype=torch.bool, device=visual.device),
            ~text_mask.bool(),
            torch.zeros((batch, 2), dtype=torch.bool, device=visual.device),
        ), dim=1)
        fused = self.fusion(tokens, src_key_padding_mask=padding)
        return self.action_head(fused[:, -1])


def behavior_cloning_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pose MSE plus gripper MSE in the dataset's normalized action domain."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 7:
        raise ValueError("Prediction and target must both be [B,7]")
    return (prediction[:, :6] - target[:, :6]).square().mean() + (
        prediction[:, 6] - target[:, 6]
    ).square().mean()
