#!/usr/bin/env python
"""
Single-file reconstruction pipeline.

Pipeline:
    Video -> DepthCrafter + DA3 -> fused depth -> DW-Mesh render video

Default behavior:
    - does NOT save intermediate depth / intrinsics / NPZ files
    - saves the final rendered video and mask video
    - optionally saves mesh ply
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'

import argparse
import gc
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.training_utils import set_seed
from imageio.v3 import imwrite
from tqdm import tqdm

import nvdiffrast.torch as dr


DEFAULT_FUSION_METHOD = "lsq"
DEFAULT_OUTPUT_NAME = "color.mp4"
DEFAULT_MASK_NAME = "mask.mp4"
DEFAULT_OUTPUT_FPS = 30


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
DA3_ROOT = REPO_ROOT / "Depth-Anything-3"
SRC_ROOT = DA3_ROOT / "src"
DEPTHCRAFTER_ROOT = WORKSPACE_ROOT / "TrajectoryCrafter" / "DepthCrafter"
DEFAULT_DA3_MODEL_DIR = DA3_ROOT / "models" / "da3_gaint_nest_1.1"

if not SRC_ROOT.exists():
    raise FileNotFoundError(f"Depth-Anything-3 src not found: {SRC_ROOT}")
if not DEPTHCRAFTER_ROOT.exists():
    raise FileNotFoundError(f"DepthCrafter root not found: {DEPTHCRAFTER_ROOT}")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(DEPTHCRAFTER_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHCRAFTER_ROOT))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model import da3 as da3_model
from depth_anything_3.model import reference_view_selector as ref_selector
from depth_anything_3.model.dinov2 import vision_transformer as da3_vit
from depth_anything_3.utils import alignment as da3_alignment
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter


def stable_least_squares_scale_scalar(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.device != b.device:
        raise ValueError(f"Device mismatch: {a.device} vs {b.device}")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Tensors must be floating point type")

    a_flat = torch.nan_to_num(a.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    b_flat = torch.nan_to_num(b.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    num = (a_flat * b_flat).sum()
    den = (b_flat * b_flat).sum().clamp_min(eps)
    return num / den


def _finite_l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def stable_select_reference_view(x: torch.Tensor, strategy: str = "middle") -> torch.Tensor:
    B, S, _, _ = x.shape
    if S <= 1:
        return torch.zeros(B, dtype=torch.long, device=x.device)
    return torch.full((B,), S // 2, dtype=torch.long, device=x.device)


def apply_da3_runtime_patches() -> None:
    da3_alignment.least_squares_scale_scalar = stable_least_squares_scale_scalar
    da3_model.least_squares_scale_scalar = stable_least_squares_scale_scalar
    ref_selector.select_reference_view = stable_select_reference_view
    da3_vit.select_reference_view = stable_select_reference_view


apply_da3_runtime_patches()


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
        device: str = "cuda:0",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        if cpu_offload is not None:
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(exc)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames: np.ndarray,
        near: float,
        far: float,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
    ):
        set_seed(seed)
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        res = res.sum(-1) / res.shape[-1]
        ori_depths = (res - res.min()) / (res.max() - res.min() + 1e-8)
        depths = torch.from_numpy(ori_depths.copy()).unsqueeze(1)
        depths *= 3900
        depths[depths < 1e-5] = 1e-5
        depths = 10000.0 / depths
        depths = depths.clip(near, far)
        return depths, ori_depths


def read_video_frames(video_path: str, process_length: int = 999999, max_res: int = 1024):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0
    while frame_count < process_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if max(h, w) > max_res:
            scale = max_res / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
        frame_count += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    return np.stack(frames, axis=0), fps


def extract_depth_from_depthcrafter(
    frames: np.ndarray,
    depth_estimater: DepthCrafterDemo,
    device: str = "cuda:0",
    near: float = 0.0001,
    far: float = 10000,
    depth_inference_steps: int = 5,
    depth_guidance_scale: float = 1.0,
    window_size: int = 110,
    overlap: int = 25,
):
    print("\nExtracting depth using DepthCrafter...")
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    depths, _ = depth_estimater.infer(
        frames,
        near,
        far,
        depth_inference_steps,
        depth_guidance_scale,
        window_size=window_size,
        overlap=overlap,
    )
    f = 500.0
    cx = depths.shape[-1] // 2
    cy = depths.shape[-2] // 2
    intrinsics = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    if not isinstance(depths, torch.Tensor):
        depths = torch.from_numpy(depths)
    return depths.to(device), intrinsics.to(device)


def extract_depth_from_da3(
    video_path: str,
    model_dir: str,
    process_res: int,
    device: str,
    use_ray_pose: bool = False,
    max_frames: int = -1,
):
    print("DA3...")

    if "cuda" in device:
        torch.cuda.empty_cache()

    try:
        model = DepthAnything3.from_pretrained(model_dir).to(device)
        device_obj = torch.device(device)
    except Exception as exc:
        print(f"DA3 fallback to CPU: {exc}")
        if "cuda" in device:
            torch.cuda.empty_cache()
        model = DepthAnything3.from_pretrained(model_dir).to("cpu")
        device_obj = torch.device("cpu")

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(frames_list) >= max_frames:
            break
    cap.release()

    print(f"Frames: {len(frames_list)}, process_res={process_res}, ref=middle")

    with torch.no_grad():
        prediction = model.inference(
            image=frames_list,
            export_dir=None,
            export_format="mini_npz",
            process_res=process_res,
            process_res_method="upper_bound_resize",
            use_ray_pose=use_ray_pose,
            ref_view_strategy="middle",
        )

    depth_maps = torch.from_numpy(prediction.depth).to(device_obj)
    if prediction.intrinsics is not None:
        intrinsics = torch.from_numpy(prediction.intrinsics[0]).float().to(device_obj)
    else:
        H, W = depth_maps.shape[1:3]
        intrinsics = torch.tensor(
            [[500.0, 0.0, W / 2.0], [0.0, 500.0, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device_obj,
        )
    del model
    if "cuda" in str(device_obj):
        torch.cuda.empty_cache()
    return depth_maps, intrinsics


def depth_fusion_lsq(depth_crafter, depth_da3):
    original_shape = depth_crafter.shape
    inv_target = 1.0 / np.clip(depth_da3.flatten(), 1e-8, None)
    inv_source = 1.0 / np.clip(depth_crafter.flatten(), 1e-8, None)
    A = np.vstack([inv_source, np.ones(len(inv_source))]).T
    x, _, _, _ = np.linalg.lstsq(A, inv_target, rcond=None)
    s, b = x[0], x[1]
    inv_dc_calibrated = s * inv_source + b
    max_inv = np.max(inv_target) * 2.0
    inv_dc_calibrated = np.clip(inv_dc_calibrated, 1e-4, max_inv)
    return (1.0 / inv_dc_calibrated).reshape(original_shape), s, b


def fuse_depths(depth_dc, depth_da3):
    print("Fusing depth...")
    if depth_dc.shape != depth_da3.shape:
        depth_da3 = F.interpolate(
            depth_da3.unsqueeze(1),
            size=depth_dc.shape[-2:],
            mode='bilinear',
            align_corners=True,
        ).squeeze(1)

    if depth_dc.shape[0] != depth_da3.shape[0]:
        min_len = min(depth_dc.shape[0], depth_da3.shape[0])
        depth_dc = depth_dc[:min_len]
        depth_da3 = depth_da3[:min_len]

    depth_dc_np = depth_dc.detach().cpu().numpy()
    depth_da3_np = depth_da3.detach().cpu().numpy()
    depth_dc_calibrated_np, _, _ = depth_fusion_lsq(depth_dc_np, depth_da3_np)
    fused = torch.from_numpy(depth_dc_calibrated_np).to(depth_dc.device).float()

    return fused


def resample_video_tensor(x: torch.Tensor, target_frames: int) -> torch.Tensor:
    if target_frames == -1 or x.shape[0] == target_frames:
        return x
    frame_indices = np.linspace(0, x.shape[0] - 1, target_frames)
    out = []
    for idx in frame_indices:
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        alpha = idx - lo
        if lo == hi:
            out.append(x[lo])
        else:
            out.append((1 - alpha) * x[lo] + alpha * x[hi])
    return torch.stack(out, dim=0)


def resample_video_np(x: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames == -1 or x.shape[0] == target_frames:
        return x
    frame_indices = np.linspace(0, x.shape[0] - 1, target_frames)
    out = []
    for idx in frame_indices:
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        alpha = idx - lo
        if lo == hi:
            out.append(x[lo])
        else:
            out.append(((1 - alpha) * x[lo] + alpha * x[hi]).astype(x.dtype))
    return np.stack(out, axis=0)


def get_rays(directions, c2w):
    rays_d = torch.einsum('kj,ij->ik', c2w[:3, :3], directions)
    rays_o = c2w[:3, 3].unsqueeze(0).expand(directions.shape[0], -1)
    return rays_o, rays_d


def get_rays_from_pose(pose, K, H, W):
    rays_screen_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack(rays_screen_coords, dim=-1).reshape(H * W, 2).to(pose)
    i, j = grid[..., 1], grid[..., 0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)
    ro, rd = get_rays(directions, pose)
    return ro.reshape(H, W, 3), rd.reshape(H, W, 3)


def getprojection(fovx, fovy, n=1.0, f=50.0, device=None):
    x = np.tan(fovx * 0.5) * n
    y = np.tan(fovy * 0.5) * n
    return torch.tensor(
        [[n / x, 0, 0, 0], [0, n / -y, 0, 0], [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)], [0, 0, -1, 0]],
        dtype=torch.float32,
        device=device,
    )


def point_to_mesh_cuda(pts, rgbs, faces, min_angle_deg=1.0, edge_threshold=0.013):
    h, w = rgbs.shape[:2]
    vertices = pts.reshape(-1, 3)
    masks = torch.ones((h, w, 1), dtype=torch.uint8, device=rgbs.device) * 255
    rgbs = torch.cat([rgbs, masks], axis=-1)
    colors = rgbs.reshape(-1, 4)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0), dim=1)

    def angle_between(v1, v2):
        cos_theta = torch.sum(v1 * v2, -1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12)
        return torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    a = angle_between(v1 - v0, v2 - v0)
    b = angle_between(v2 - v1, v0 - v1)
    c = angle_between(v0 - v2, v1 - v2)
    min_angles = torch.minimum(torch.minimum(a, b), c)
    valid_faces = min_angles >= min_angle_deg
    z_range = (vertices[:, 2].max() - vertices[:, 2].min()).clamp_min(1e-8)
    z01, z12, z20 = torch.abs((v0 - v1)[:, 2]), torch.abs((v1 - v2)[:, 2]), torch.abs((v2 - v0)[:, 2])
    y01, y12, y20 = torch.abs((v0 - v1)[:, 1]), torch.abs((v1 - v2)[:, 1]), torch.abs((v2 - v0)[:, 1])
    x01, x12, x20 = torch.abs((v0 - v1)[:, 0]), torch.abs((v1 - v2)[:, 0]), torch.abs((v2 - v0)[:, 0])
    proj_max = torch.maximum(torch.maximum(torch.maximum(x01, x12), x20), torch.maximum(torch.maximum(y01, y12), torch.maximum(y20, torch.maximum(z01, torch.maximum(z12, z20)))))
    valid_faces_final = valid_faces & (proj_max / z_range < edge_threshold)
    invalid_faces = faces[~valid_faces_final]
    if len(invalid_faces) > 0:
        invalid_vertex_indices = torch.unique(invalid_faces.flatten())
        colors[invalid_vertex_indices] = 0
    return vertices, faces, colors, face_normals


def render_nvdiffrast(glctx, vertices, faces, colors, proj, poses, h, w):
    if faces.shape[0] == 0:
        raise RuntimeError(f"faces must have shape [>0, 3], got {faces.shape}")

    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1], device=pos.device)], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, h, w):
        pos_clip = transform_pos(mtx, pos)
        rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[h, w])
        color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
        color = dr.antialias(color, rast_out, pos_clip, pos_idx)
        return color

    poses = poses.clone()
    poses[0, :] *= -1
    poses[1, :] *= -1
    poses[2, :] *= -1
    mvp = proj @ poses
    return render(glctx, mvp, vertices, faces, colors, faces, h, w)


def get_camera_pose(eye, center):
    up = np.array((0, 1, 0), dtype=np.float32)

    def normalize(v):
        norm = np.linalg.norm(v)
        return v if norm < 1e-8 else v / norm

    forward = normalize(center - eye)
    right = normalize(np.cross(forward, up))
    new_up = normalize(np.cross(right, forward))
    view = np.zeros((4, 4), dtype=np.float32)
    view[0, 0:3] = right
    view[1, 0:3] = new_up
    view[2, 0:3] = forward
    view[0:3, 3] = -np.array([np.dot(right, eye), np.dot(new_up, eye), np.dot(forward, eye)])
    view[3, 3] = 1.0
    return torch.from_numpy(view)


def fixed_camera_traj(n_frames, depth_src, view_type="front", angle=30, depth_min=None, zoom_out_factor=1.0):
    radius = depth_min * zoom_out_factor
    eye = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center = np.array([0.0, 0.0, radius], dtype=np.float32)
    if view_type == "front":
        eye[2] = 0.0
        center[2] = radius
    elif view_type == "left":
        angle_rad = np.radians(-angle)
        eye[0] = np.sin(angle_rad) * radius
        eye[2] = radius - radius * np.abs(np.cos(angle_rad))
        center[2] = radius
    elif view_type == "right":
        angle_rad = np.radians(angle)
        eye[0] = np.sin(angle_rad) * radius
        eye[2] = radius - radius * np.abs(np.cos(angle_rad))
        center[2] = radius
    elif view_type == "angle":
        angle_rad = np.radians(angle)
        eye[0] = np.sin(angle_rad) * radius
        eye[2] = radius - radius * np.abs(np.cos(angle_rad))
        center[2] = radius
    camera_pose = get_camera_pose(eye, center)
    return torch.stack([camera_pose for _ in range(n_frames)], 0).to(depth_src.device)


def generate_faces(H, W, device):
    idx = np.arange(H * W).reshape(H, W)
    faces = torch.from_numpy(np.concatenate([
        np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=-1),
        np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()], axis=-1),
    ], axis=0)).int().to(device)
    return faces[:, [1, 0, 2]]


def render_dwmesh(depth_fused, frames_rgb, intrinsics, args, output_dir):
    print("Rendering...")
    device = args.device
    n_frames = depth_fused.shape[0]
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    if depth_fused.ndim == 4 and depth_fused.shape[1] == 1:
        depth_fused = depth_fused.squeeze(1)

    glctx = dr.RasterizeCudaContext(device=device)
    H, W = depth_fused.shape[1:3]

    frames_H, frames_W = frames_rgb.shape[1:3]
    if (frames_H, frames_W) != (H, W):
        frames_rgb = F.interpolate(
            frames_rgb.permute(0, 3, 1, 2),
            size=(H, W),
            mode='bilinear',
            align_corners=True,
        ).permute(0, 2, 3, 1)

    fov_y = 2 * math.atan2(H, 2 * intrinsics[1, 1].item())
    fov_x = 2 * math.atan2(W, 2 * intrinsics[0, 0].item())
    fx = fy = 0.5 * H / math.tan(fov_y / 2)
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=device)

    pose = torch.eye(4, device=device)
    ro_src, rd_src = get_rays_from_pose(pose, K, H, W)
    proj = getprojection(fov_x, fov_y, n=1e-3, f=1e3, device=device)

    depth_min = depth_fused[0].min().item() + args.depth_offset
    camera_poses = fixed_camera_traj(
        n_frames=n_frames,
        depth_src=depth_fused,
        view_type=args.view_type,
        angle=args.angle,
        depth_min=depth_min,
        zoom_out_factor=args.zoom_out_factor,
    )

    depth_src = depth_fused.clone()
    depth_src[:, 0, :] = args.edge_depth_value
    depth_src[:, -1, :] = args.edge_depth_value
    depth_src[:, :, 0] = args.edge_depth_value
    depth_src[:, :, -1] = args.edge_depth_value
    depth_src = depth_src.unsqueeze(-1)

    video = []
    video_mask = []

    for idx in tqdm(range(n_frames)):
        pts_color = frames_rgb[idx]
        pts_xyz = depth_src[idx] * rd_src + ro_src
        faces = generate_faces(H, W, device)
        vertices, new_faces, colors, _ = point_to_mesh_cuda(
            pts_xyz,
            pts_color,
            faces,
            min_angle_deg=args.min_angle_deg,
            edge_threshold=args.edge_threshold,
        )

        if idx == 0 and args.save_mesh:
            import trimesh
            mesh = trimesh.Trimesh(vertices.cpu().numpy(), new_faces.cpu().numpy())
            mesh.visual.vertex_colors = colors.cpu().numpy().astype(np.uint8)[:, :3]
            mesh.export(os.path.join(output_dir, "DW-Mesh.ply"))

        if idx == 0:
            video.append(pts_color.cpu().numpy().astype(np.uint8))
            video_mask.append(np.ones((H, W), dtype=np.uint8) * 255)
            continue

        img = render_nvdiffrast(glctx, vertices, new_faces, colors, proj, camera_poses[idx], H, W)[0]
        mask = img[..., 3:]
        mask[mask > args.mask_threshold] = 255
        mask[mask <= args.mask_threshold] = 0
        img[..., 3:] = mask
        img[..., :3] = img[..., :3] * (mask / 255)
        video.append(img[..., :3].cpu().numpy().astype(np.uint8))
        video_mask.append(mask.squeeze(-1).cpu().numpy().astype(np.uint8))

    video = np.stack(video, axis=0).astype(np.uint8)
    color_path = os.path.join(output_dir, DEFAULT_OUTPUT_NAME)
    imwrite(color_path, video[..., :3], fps=DEFAULT_OUTPUT_FPS)
    mask_video = np.stack(video_mask, axis=0).astype(np.uint8)
    mask_path = os.path.join(output_dir, DEFAULT_MASK_NAME)
    imwrite(mask_path, mask_video, fps=DEFAULT_OUTPUT_FPS)


def main():
    parser = argparse.ArgumentParser(description="Single-file recon pipeline (DepthCrafter + DA3 + DW-Mesh)")
    parser.add_argument("--input_video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--da3_model_dir", type=str, default=str(DEFAULT_DA3_MODEL_DIR), help="DA3 模型目录")
    parser.add_argument("--target_n_frames", type=int, default=49, help="目标帧数")
    parser.add_argument("--max_res", type=int, default=320, help="输入帧最大分辨率")
    parser.add_argument("--da3_process_res", type=int, default=504, help="DA3 处理分辨率")
    parser.add_argument("--rayhead", action="store_true", default=False, help="启用 DA3 ray head pose")
    parser.add_argument("--dc_steps", type=int, default=5, help="DepthCrafter 推理步数")
    parser.add_argument("--view_type", type=str, default="front", choices=["front", "left", "right", "angle"], help="渲染固定视角")
    parser.add_argument("--angle", type=float, default=30.0, help="固定视角角度")
    parser.add_argument("--zoom_out_factor", type=float, default=1.3, help="缩放因子")
    parser.add_argument("--min_angle_deg", type=float, default=3.0, help="mesh 面片最小角")
    parser.add_argument("--edge_threshold", type=float, default=0.0008, help="mesh 边界阈值")
    parser.add_argument("--edge_depth_value", type=float, default=100.0, help="边界深度值")
    parser.add_argument("--mask_threshold", type=int, default=125, help="mask 阈值")
    parser.add_argument("--depth_offset", type=float, default=0.15, help="渲染半径偏移")
    parser.add_argument("--save_mesh", action="store_true", help="同时保存第一帧 mesh")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input: {args.input_video}")
    print(f"Output: {args.output_dir}")

    process_len = args.target_n_frames
    frames_np, _ = read_video_frames(args.input_video, process_length=process_len, max_res=args.max_res)
    print(f"Frames: {frames_np.shape[0]}")

    dc_model = None
    try:
        print("DepthCrafter...")
        dc_model = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model",
            device=args.device,
        )
        depth_dc, intrinsics_dc = extract_depth_from_depthcrafter(
            frames_np,
            dc_model,
            device=args.device,
            depth_inference_steps=args.dc_steps,
        )
    finally:
        if dc_model is not None:
            del dc_model
        gc.collect()
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    depth_da3, intrinsics_da3 = extract_depth_from_da3(
        args.input_video,
        args.da3_model_dir,
        args.da3_process_res,
        args.device,
        use_ray_pose=args.rayhead,
        max_frames=process_len,
    )

    depth_dc = resample_video_tensor(depth_dc, args.target_n_frames)
    depth_da3 = resample_video_tensor(depth_da3, args.target_n_frames)
    frames_np = resample_video_np(frames_np, args.target_n_frames)

    if depth_dc.device != depth_da3.device:
        depth_da3 = depth_da3.to(depth_dc.device)
    if intrinsics_da3.device != depth_dc.device:
        intrinsics_da3 = intrinsics_da3.to(depth_dc.device)

    depth_fused = fuse_depths(depth_dc, depth_da3)

    final_H, final_W = depth_fused.shape[-2:]
    da3_H, da3_W = depth_da3.shape[-2:]
    if (final_H, final_W) != (da3_H, da3_W):
        scale_x = final_W / da3_W
        scale_y = final_H / da3_H
        intrinsics_final = intrinsics_da3.clone()
        intrinsics_final[0, 0] *= scale_x
        intrinsics_final[1, 1] *= scale_y
        intrinsics_final[0, 2] *= scale_x
        intrinsics_final[1, 2] *= scale_y
    else:
        intrinsics_final = intrinsics_da3

    frames_rgb = torch.from_numpy(frames_np).float().to(args.device)
    render_dwmesh(depth_fused, frames_rgb, intrinsics_final, args, args.output_dir)

    print(f"Color: {os.path.join(args.output_dir, DEFAULT_OUTPUT_NAME)}")
    print(f"Mask: {os.path.join(args.output_dir, DEFAULT_MASK_NAME)}")
    if args.save_mesh:
        print(f"Mesh: {os.path.join(args.output_dir, 'DW-Mesh.ply')}")


if __name__ == "__main__":
    main()
