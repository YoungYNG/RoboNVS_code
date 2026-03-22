"""
Camera Encoder with Reference Frame Fusion (Plan A)
This version uses channel concatenation to fuse video, mask, and reference frame.
"""
import torch
import torch.nn as nn


class CamVidEncoderFusion(nn.Module):
    """
    Camera encoder with reference frame support using direct concatenation fusion.

    Architecture:
        video_latent [B, 16, T, H, W] \
        mask_latent  [B, 16, T, H, W]  -> Concat -> Conv3d -> [B, out_channels, T, H', W']
        ref_latent   [B, 16, T, H, W] /
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 1024,
        out_channels: int = 5120,
        use_ref_frame: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_ref_frame = use_ref_frame

        # Input channels calculation
        # Original: video(16) + mask(16) = 32
        # With ref: video(16) + mask(16) + ref(16) = 48
        encoder_in_channels = in_channels * 2
        if use_ref_frame:
            encoder_in_channels = in_channels * 3

        self.latent_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(encoder_in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        )

        self.latent_patch_embedding = torch.nn.Conv3d(
            hidden_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )

        # Initialize output layer to zero (important for training stability)
        nn.init.zeros_(self.latent_patch_embedding.weight)
        nn.init.zeros_(self.latent_patch_embedding.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nn.Module):
            module.gradient_checkpointing = value

    def forward(self, video_latent, mask_latent, ref_frame_latent=None) -> torch.Tensor:
        """
        Forward pass with precomputed latents.

        Args:
            video_latent: [B, 16, T, H, W] - Precomputed video latent (with holes)
            mask_latent: [B, 16, T, H, W] - Precomputed mask latent
            ref_frame_latent: [B, 16, 1, H, W] - Precomputed reference frame latent (optional)

        Returns:
            ray_latent: [B, out_channels, T, H', W'] - Camera embedding
        """
        B, C, T, H, W = video_latent.shape

        if self.use_ref_frame and ref_frame_latent is not None:
            # Expand reference frame to all time steps
            if ref_frame_latent.shape[2] == 1:
                # ref_frame_latent: [B, 16, 1, H, W] -> [B, 16, T, H, W]
                ref_latent_expanded = ref_frame_latent.expand(-1, -1, T, -1, -1)
            else:
                ref_latent_expanded = ref_frame_latent

            # Three-way concatenation: video + mask + ref
            latent = torch.cat([video_latent, mask_latent, ref_latent_expanded], dim=1)  # [B, 48, T, H, W]
        else:
            # Two-way concatenation (original): video + mask
            latent = torch.cat([video_latent, mask_latent], dim=1)  # [B, 32, T, H, W]

        # Process through encoder
        latent = self.latent_encoder(latent)  # [B, 1024, T, H, W]
        latent = self.latent_patch_embedding(latent)  # [B, 5120, T, H', W']

        return latent


def prepare_camera_embeds_fusion(
    camera_encoder,
    video_latent,
    mask_latent,
    ref_frame_latent=None
) -> torch.Tensor:
    """
    Prepare camera embeddings with optional reference frame (precomputed latents version).

    Args:
        camera_encoder: CamVidEncoderFusion instance
        video_latent: [B, 16, T, H, W] - Precomputed video latent
        mask_latent: [B, 16, T, H, W] - Precomputed mask latent
        ref_frame_latent: [B, 16, 1, H, W] - Precomputed reference frame latent (optional)

    Returns:
        ray_latent: [B, out_channels, T, H', W']
    """
    ray_latent = camera_encoder(video_latent, mask_latent, ref_frame_latent)
    return ray_latent
