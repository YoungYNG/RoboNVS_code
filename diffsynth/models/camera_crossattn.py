"""
Camera Encoder with Reference Frame Cross-Attention (Plan C)
This version uses cross-attention to model cross-view correspondence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewAttention3D(nn.Module):
    """
    3D Cross-View Attention Module.

    This module allows the video features to attend to reference frame features,
    learning cross-view correspondence in a data-driven way.
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Query from video features
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        # Key and Value from reference features
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        # Output projection
        self.out = nn.Conv3d(channels, channels, kernel_size=1)

        # Initialize output to zero for training stability
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, video_feat, ref_feat, mask=None):
        """
        Args:
            video_feat: [B, C, T, H, W] - Video features (query)
            ref_feat: [B, C, T, H, W] - Reference features (key/value)
            mask: [B, 1, T, H, W] - Optional mask (1=valid, 0=hole)

        Returns:
            out: [B, C, T, H, W] - Attended features
        """
        B, C, T, H, W = video_feat.shape

        # Generate Q, K, V
        q = self.q(video_feat)  # [B, C, T, H, W]
        k = self.k(ref_feat)    # [B, C, T, H, W]
        v = self.v(ref_feat)    # [B, C, T, H, W]

        # Reshape to multi-head format
        # [B, C, T, H, W] -> [B, num_heads, head_dim, T*H*W]
        q = q.view(B, self.num_heads, self.head_dim, T * H * W)
        k = k.view(B, self.num_heads, self.head_dim, T * H * W)
        v = v.view(B, self.num_heads, self.head_dim, T * H * W)

        # Transpose for attention: [B, num_heads, T*H*W, head_dim]
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # Compute attention scores: [B, num_heads, T*H*W, T*H*W]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention weights
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, num_heads, T*H*W, head_dim]

        # Reshape back: [B, num_heads, T*H*W, head_dim] -> [B, C, T, H, W]
        out = out.transpose(2, 3).contiguous()
        out = out.view(B, C, T, H, W)

        # Output projection
        out = self.out(out)

        # Optional: Apply mask to only use attention in hole regions
        if mask is not None:
            # mask: [B, 1, T, H, W], 1=valid, 0=hole
            # We want to use attention primarily in hole regions (mask=0)
            # So: output = (1 - mask) * attention_result + mask * video_feat
            # But for simplicity, we just modulate the attention output
            out = out * (1 - mask)

        return out


class CamVidEncoderCrossAttn(nn.Module):
    """
    Camera encoder with cross-view attention for reference frame fusion.

    Architecture:
        1. Encode video+mask features
        2. Encode reference frame features
        3. Apply cross-view attention
        4. Fuse and output
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 1024,
        out_channels: int = 5120,
        num_attn_heads: int = 8,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_attn_heads = num_attn_heads

        # Video branch: encode video + mask
        self.video_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
        )

        # Reference branch: encode reference frame
        self.ref_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
        )

        # Cross-view attention
        self.cross_view_attn = CrossViewAttention3D(hidden_channels, num_heads=num_attn_heads)

        # Fusion layer
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
        )

        # Patch embedding (spatial downsampling)
        self.latent_patch_embedding = torch.nn.Conv3d(
            hidden_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )

        # Initialize output layer to zero
        nn.init.zeros_(self.latent_patch_embedding.weight)
        nn.init.zeros_(self.latent_patch_embedding.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nn.Module):
            module.gradient_checkpointing = value

    def forward(self, video_latent, mask_latent, ref_frame_latent=None) -> torch.Tensor:
        """
        Forward pass with cross-view attention.

        Args:
            video_latent: [B, 16, T, H, W] - Precomputed video latent (with holes)
            mask_latent: [B, 16, T, H, W] - Precomputed mask latent
            ref_frame_latent: [B, 16, 1, H, W] - Precomputed reference frame latent (optional)

        Returns:
            ray_latent: [B, out_channels, T, H', W'] - Camera embedding
        """
        B, C, T, H, W = video_latent.shape

        # Encode video + mask
        video_mask_concat = torch.cat([video_latent, mask_latent], dim=1)  # [B, 32, T, H, W]
        video_feat = self.video_encoder(video_mask_concat)  # [B, 1024, T, H, W]

        if ref_frame_latent is not None:
            # Expand reference frame to all time steps
            if ref_frame_latent.shape[2] == 1:
                ref_latent_expanded = ref_frame_latent.expand(-1, -1, T, -1, -1)  # [B, 16, T, H, W]
            else:
                ref_latent_expanded = ref_frame_latent

            # Encode reference frame
            ref_feat = self.ref_encoder(ref_latent_expanded)  # [B, 1024, T, H, W]

            # Cross-view attention: video attends to reference
            # Use first channel of mask_latent as spatial mask (1=valid, 0=hole)
            spatial_mask = (mask_latent[:, :1] > 0).float()  # [B, 1, T, H, W]
            attended_ref = self.cross_view_attn(video_feat, ref_feat, mask=None)

            # Fuse: add attended reference features to video features
            fused = video_feat + attended_ref
        else:
            # No reference frame, just use video features
            fused = video_feat

        # Final processing
        latent = self.fusion(fused)  # [B, 1024, T, H, W]
        latent = self.latent_patch_embedding(latent)  # [B, 5120, T, H', W']

        return latent


def prepare_camera_embeds_crossattn(
    camera_encoder,
    video_latent,
    mask_latent,
    ref_frame_latent=None
) -> torch.Tensor:
    """
    Prepare camera embeddings with cross-attention fusion (precomputed latents version).

    Args:
        camera_encoder: CamVidEncoderCrossAttn instance
        video_latent: [B, 16, T, H, W] - Precomputed video latent
        mask_latent: [B, 16, T, H, W] - Precomputed mask latent
        ref_frame_latent: [B, 16, 1, H, W] - Precomputed reference frame latent (optional)

    Returns:
        ray_latent: [B, out_channels, T, H', W']
    """
    ray_latent = camera_encoder(video_latent, mask_latent, ref_frame_latent)
    return ray_latent
