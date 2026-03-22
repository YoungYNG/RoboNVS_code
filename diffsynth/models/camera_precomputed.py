"""
Camera encoder for precomputed VAE latents.
This version skips VAE encoding and directly uses precomputed latents for faster training.
"""
import torch
import torch.nn as nn


class CamVidEncoderPrecomputed(nn.Module):
    """
    A VAE model for encoding camera information and video features.
    This version works with precomputed VAE latents (no VAE encoding during forward pass).
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 1024,
        out_channels: int = 5120,
    ) -> None:
        super().__init__()

        self.latent_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        )
        self.latent_patch_embedding = torch.nn.Conv3d(hidden_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        nn.init.zeros_(self.latent_patch_embedding.weight)
        nn.init.zeros_(self.latent_patch_embedding.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nn.Module):
            module.gradient_checkpointing = value

    def forward(self, video_latent, mask_latent) -> torch.Tensor:
        """
        Forward pass with precomputed latents (no VAE encoding).

        Args:
            video_latent: Precomputed VAE latent for color video
            mask_latent: Precomputed VAE latent for mask video

        Returns:
            torch.Tensor: Camera embeddings
        """
        # Concatenate precomputed latents (no VAE encoding needed)
        latent = torch.cat([video_latent, mask_latent], dim=1)
        latent = self.latent_encoder(latent)
        latent = self.latent_patch_embedding(latent)
        return latent


def prepare_camera_embeds_precomputed(
    camera_encoder,
    video_latent,
    mask_latent
) -> torch.Tensor:
    """
    Prepare camera embeddings using precomputed latents (no VAE).

    Args:
        camera_encoder: Camera encoder model (CamVidEncoderPrecomputed)
        video_latent: Precomputed VAE latent for color video
        mask_latent: Precomputed VAE latent for mask video

    Returns:
        torch.Tensor: Camera embeddings
    """
    ray_latent = camera_encoder(video_latent, mask_latent)
    return ray_latent
