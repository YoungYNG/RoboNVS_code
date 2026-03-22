#!/usr/bin/env python
"""
深度融合脚本：结合 DepthCrafter 和 DA3 生成融合深度
Pipeline: Video -> Fused Depth (DC+DA3) -> NPZ
输出: 融合深度数据 + 内参 + 外参 + 可视化视频

深度融合实现：
- 支持多种融合方法：LSQ、Average、Weighted、Max、Geometric
- Geometric 融合基于论文 3.2 节：利用内外参进行投影空间的深度融合
  - 流程：反投影 → 3D变换 → 重投影 → 采样
  - 需要：DA3 外参 (--rayhead 可提高外参精度)
"""

import os
# 设置离线模式（必须在 import torch 之前）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import sys
from imageio.v3 import imwrite

# 强制优先使用当前仓库的源码，避免误导入 conda 环境里绑定到其他目录的 editable 包
repo_root = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.join(repo_root, "src")
if os.path.isdir(src_root) and src_root not in sys.path:
    sys.path.insert(0, src_root)

# 导入 Depth-Anything-3 相关模块
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: depth_anything_3 not installed. Please install it first.")
    sys.exit(1)

# ==================== 配置 EX-4D 导入路径 ====================
workspace_root = os.path.dirname(repo_root)
workspace_parent = os.path.dirname(workspace_root)
ex4d_candidates = [
    workspace_root,
    os.path.join(workspace_parent, "EX-4D"),
    os.path.join(workspace_parent, "EX-4D_open"),
]
ex4d_root = next((path for path in ex4d_candidates if os.path.isdir(path)), None)

if ex4d_root is None:
    print("Error: EX-4D root not found. Expected one of:")
    for path in ex4d_candidates:
        print(f"  - {path}")
    sys.exit(1)

depthcrafter_root = os.path.join(ex4d_root, "DepthCrafter")
if os.path.isdir(depthcrafter_root) and depthcrafter_root not in sys.path:
    sys.path.insert(0, depthcrafter_root)
if ex4d_root not in sys.path:
    sys.path.insert(0, ex4d_root)

from utils.depth_utils import DepthCrafterDemo
from utils.dc_utils import read_video_frames


# ==================== 深度融合核心函数 ====================

def unproject_to_3d(depth, intrinsics, extrinsics):
    """
    3.2节：反投影 (Unprojection)
    将 2D 深度图投影回 3D 世界坐标

    Args:
        depth: 深度图 (H, W) 或 (B, H, W)
        intrinsics: 内参矩阵 (3, 3) 或 (B, 3, 3)
        extrinsics: 外参矩阵 (4, 4) 或 (B, 4, 4)

    Returns:
        world_coords: 世界坐标系下的 3D 点 (B, H, W, 3)
    """
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth).float()
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float()
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics).float()

    # 处理维度
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)  # (1, H, W)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)  # (1, 3, 3)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0)  # (1, 4, 4)

    B, H, W = depth.shape
    device = depth.device

    # 1. 创建像素坐标网格
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(H, 1).expand(H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, W).expand(H, W)
    ones = torch.ones_like(x_coords)

    # 2. 齐次坐标 [u, v, 1]
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=-1)  # (H, W, 3)

    # 3. 利用内参反投影到相机坐标系
    # [X_c, Y_c, Z_c] = Z_d * K^{-1} * [u, v, 1]
    K_inv = torch.inverse(intrinsics[0])  # (3, 3)
    camera_coords = torch.einsum('ij,hwj->hwi', K_inv, pixel_coords)  # (H, W, 3)

    # 缩放深度
    camera_coords = camera_coords * depth[0].unsqueeze(-1)  # (H, W, 3)

    # 4. 齐次化：从相机坐标转换到世界坐标
    # [X_w, Y_w, Z_w, 1] = extrinsics^{-1} * [X_c, Y_c, Z_c, 1]
    camera_coords_homo = torch.cat(
        [camera_coords, torch.ones((H, W, 1), device=device)], dim=-1
    )  # (H, W, 4)

    extrinsics_inv = torch.inverse(extrinsics[0])  # (4, 4)  # c2w矩阵逆
    world_coords = torch.einsum('ij,hwj->hwi', extrinsics_inv, camera_coords_homo)  # (H, W, 4)
    world_coords = world_coords[..., :3]  # (H, W, 3)

    return world_coords.unsqueeze(0)  # (1, H, W, 3)


def project_to_2d(world_coords, intrinsics, extrinsics):
    """
    3.2节：重投影 (Reprojection)
    将 3D 世界坐标投影到目标相机的 2D 平面

    Args:
        world_coords: 世界坐标 (1, H, W, 3)
        intrinsics: 目标相机内参 (3, 3)
        extrinsics: 目标相机外参 (4, 4)

    Returns:
        target_coords: 归一化坐标 (1, H, W, 2) [-1, 1] 范围
    """
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float()
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics).float()

    device = world_coords.device

    # 1. 世界坐标 → 相机坐标：利用外参
    world_coords_homo = torch.cat(
        [world_coords, torch.ones_like(world_coords[..., :1])], dim=-1
    )  # (1, H, W, 4)

    extrinsics = extrinsics.to(device)
    camera_coords_homo = torch.einsum('ij,bhwj->bhwi', extrinsics, world_coords_homo)
    camera_coords = camera_coords_homo[..., :3]  # (1, H, W, 3)

    # 2. 相机坐标 → 图像坐标：利用内参
    intrinsics = intrinsics.to(device)
    pixel_homo = torch.einsum('ij,bhwj->bhwi', intrinsics, camera_coords)  # (1, H, W, 3)
    pixel_coords = pixel_homo[..., :2] / pixel_homo[..., 2:3]  # (1, H, W, 2)

    # 3. 归一化到 [-1, 1] 范围（用于 grid_sample）
    _, _, H, W = world_coords.shape
    pixel_coords[..., 0] = 2.0 * pixel_coords[..., 0] / (W - 1) - 1.0
    pixel_coords[..., 1] = 2.0 * pixel_coords[..., 1] / (H - 1) - 1.0

    return pixel_coords


def get_warped_depth_guidance(source_depth, K_source, P_source, K_target, P_target):
    """
    论文 3.2 节核心：生成扭曲深度图 (Warped Depth)
    这是深度融合的关键输入。

    深度融合流程：
    1. 反投影：利用源视频的内参 K_source 和外参 P_source，将 2D 深度图变换到 3D 世界空间
    2. 重投影：利用目标相机的内参 K_target 和外参 P_target，投影回 2D 平面
    3. 结果：得到扭曲深度图 D_r，可用于视觉引导

    Args:
        source_depth: 源视频深度图 (H, W) 或 (B, H, W)
        K_source: 源相机内参 (3, 3)
        P_source: 源相机外参 (4, 4)
        K_target: 目标相机内参 (3, 3)
        P_target: 目标相机外参 (4, 4)

    Returns:
        warped_depth: 扭曲后的深度图 (1, 1, H, W)
        validity_mask: 有效性掩码 (1, 1, H, W) - 指示重投影是否有效
    """
    device = source_depth.device if isinstance(source_depth, torch.Tensor) else 'cpu'

    # 步骤 A: 反投影到 3D 世界坐标
    print(f"  [深度融合] 步骤 A: 反投影到 3D 世界空间...")
    world_coords = unproject_to_3d(source_depth, K_source, P_source)
    world_coords = world_coords.to(device)
    print(f"    世界坐标范围: [{world_coords.min():.4f}, {world_coords.max():.4f}]")

    # 步骤 B: 重投影到目标相机平面
    print(f"  [深度融合] 步骤 B: 重投影到目标相机视角...")
    target_pixel_coords = project_to_2d(world_coords, K_target, P_target)
    target_pixel_coords = target_pixel_coords.to(device)

    # 步骤 C: 使用 grid_sample 获取扭曲的深度图
    print(f"  [深度融合] 步骤 C: 采样扭曲深度...")
    source_depth_input = source_depth.unsqueeze(0).unsqueeze(0) if source_depth.ndim == 2 else source_depth.unsqueeze(1)
    source_depth_input = source_depth_input.to(device)

    warped_depth = F.grid_sample(
        source_depth_input,
        target_pixel_coords,
        mode='bilinear',
        padding_mode='zeros',  # 超出边界的像素为 0
        align_corners=True
    )  # (1, 1, H, W)

    # 生成有效性掩码：指示重投影坐标是否在图像范围内
    valid_mask = (target_pixel_coords[..., 0] >= -1.0) & (target_pixel_coords[..., 0] <= 1.0) & \
                 (target_pixel_coords[..., 1] >= -1.0) & (target_pixel_coords[..., 1] <= 1.0)
    valid_mask = valid_mask.unsqueeze(1).float()  # (1, 1, H, W)

    print(f"    扭曲深度范围: [{warped_depth.min():.4f}, {warped_depth.max():.4f}]")
    print(f"    有效像素比例: {valid_mask.mean():.2%}")

    return warped_depth, valid_mask


def depth_injection(noise_latent, warped_depth, depth_scale=1.0):
    """
    论文 3.2 节末尾：深度注入 (Depth Injection)
    将扭曲的深度图与噪声特征融合

    论文原文：
    "We inject the viewpoint guidance (i.e., the warped depth) into
    the reverse diffusion process by adding it with the noisy latents"

    Args:
        noise_latent: 噪声特征/Latent (B, C, H, W) 来自扩散模型
        warped_depth: 扭曲深度图 (B, 1, H, W)
        depth_scale: 深度权重因子（控制融合强度）

    Returns:
        fused_latent: 融合后的特征 (B, C, H, W)
    """
    B, C, H, W = noise_latent.shape

    # 1. 将深度图重采样到噪声特征的空间分辨率
    if warped_depth.shape[2:] != (H, W):
        warped_depth = F.interpolate(
            warped_depth,
            size=(H, W),
            mode='bilinear',
            align_corners=True
        )

    # 2. 扩展深度图维度以匹配特征维度（可选：使用线性投影）
    # 简单版本：直接广播
    # 高级版本：使用可学习的线性投影
    depth_feature = warped_depth.expand(B, C, H, W)  # (B, C, H, W)

    # 3. 融合：相加
    # 论文中明确指出使用加法作为融合操作
    fused_latent = noise_latent + depth_scale * depth_feature

    return fused_latent


# ==================== 辅助函数 ====================

def get_video_info(video_path):
    """获取视频的帧率和总帧数"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def extract_depth_from_depthcrafter(frames, depth_estimater, device='cuda:0',
                                   near=0.0001, far=10000,
                                   depth_inference_steps=5,
                                   depth_guidance_scale=1.0,
                                   window_size=110, overlap=25):
    """使用 DepthCrafter 模型提取深度数据"""
    print(f"\nExtracting depth using DepthCrafter...")
    print(f"  Device: {device}")

    # 设置当前 CUDA 设备（确保 infer 使用正确的设备）
    if 'cuda' in device:
        gpu_id = int(device.split(':')[1])
        torch.cuda.set_device(gpu_id)
        print(f"  Set CUDA device to GPU {gpu_id}")

    # 确保输入是 float32 且在 [0,1]
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    depths, _ = depth_estimater.infer(
        frames,
        near, far,
        depth_inference_steps,
        depth_guidance_scale,
        window_size=window_size,
        overlap=overlap
    )

    print(f"  DepthCrafter raw output shape: {depths.shape}")
    print(f"  Raw range: [{depths.min():.4f}, {depths.max():.4f}]")

    # DepthCrafter 通常输出视差而不是深度，检查是否需要转换
    # 如果值在 0-10 之间，可能是视差；如果值都很小（<1），可能是倒数视差
    if depths.mean() > 1.0:
        print(f"  ℹ️  Detected large values, treating as disparity/inverse depth")
        print(f"  ℹ️  Keeping as-is for test_demo.py (它会自己处理)")
    else:
        print(f"  ℹ️  Detected small values (<1), likely already depth")

    f = 500
    cx = depths.shape[-1] // 2
    cy = depths.shape[-2] // 2
    intrinsics = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    print(f"  DepthCrafter depth shape: {depths.shape}")
    print(f"  Depth range: [{depths.min():.4f}, {depths.max():.4f}]")

    return depths, intrinsics


def extract_depth_from_da3(
    video_path,
    model_dir,
    process_res,
    device,
    use_ray_pose=False,
    max_frames=-1,
    ref_view_strategy="middle",
):
    """使用 Depth-Anything-3 模型提取深度数据和相机内参"""
    print(f"\nExtracting depth using Depth-Anything-3...")
    print(f"  Loading DA3 model from {model_dir}...")

    # 清理 GPU 缓存
    if 'cuda' in device:
        torch.cuda.empty_cache()

    # 加载模型 - 使用简单直接的方式，遵循 process_video_to_npz.py 的模式
    try:
        model = DepthAnything3.from_pretrained(model_dir).to(device)
        print(f"  ✓ DA3 model loaded on {device}")
        device_obj = torch.device(device)
    except (RuntimeError, Exception) as e:
        print(f"  ⚠️  Failed to load on {device}: {e}")
        print(f"     Falling back to CPU...")
        try:
            if 'cuda' in device:
                torch.cuda.empty_cache()
            model = DepthAnything3.from_pretrained(model_dir).to('cpu')
            device_obj = torch.device('cpu')
            print(f"  ✓ DA3 model loaded on CPU")
        except Exception as e2:
            print(f"  ❌ Failed to load DA3 model: {e2}")
            raise

    # 2. 从视频中提取帧 (RGB 格式)
    print(f"Loading frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()

    N = len(frames)
    print(f"Loaded {N} frames from video")

    # 3. 运行 DA3 推理 - 遵循 process_video_to_npz.py 的模式
    print(f"Running DA3 inference (process_res={process_res})...")
    print(f"  Reference view strategy: {ref_view_strategy}")
    with torch.no_grad():
        prediction = model.inference(
            image=frames,
            export_dir=None,
            export_format="mini_npz",
            process_res=process_res,
            process_res_method="upper_bound_resize",
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        )
    print(f"  ✓ DA3 inference completed")

    # 4. 提取深度图 (N, H, W)
    depth_maps = torch.from_numpy(prediction.depth).to(device_obj)
    print(f"  Depth shape: {depth_maps.shape}")
    print(f"  Depth range: [{depth_maps.min():.4f}, {depth_maps.max():.4f}]")

    # 5. 提取相机内参
    if prediction.intrinsics is not None:
        intrinsics = torch.from_numpy(prediction.intrinsics[0]).float().to(device_obj)
        print(f"  Camera intrinsics: fx={intrinsics[0,0]:.2f}, fy={intrinsics[1,1]:.2f}, cx={intrinsics[0,2]:.2f}, cy={intrinsics[1,2]:.2f}")
    else:
        H, W = depth_maps.shape[1:3]
        focal_length = 500.0
        intrinsics = torch.tensor([
            [focal_length, 0.0, W / 2.0],
            [0.0, focal_length, H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device_obj)
        print(f"  ⚠️  Using default intrinsics (focal_length=500)")

    # 6. 提取 FOV 信息 (可选)
    fov_values = None
    if hasattr(prediction, 'fov') and prediction.fov is not None:
        fov_values = prediction.fov
        print(f"  ✓ FOV extracted: shape {fov_values.shape}")
    else:
        print(f"  ℹ️  FOV not provided by DA3")

    # 7. 提取相机外参 (可选)
    extrinsics = None
    if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
        extrinsics = prediction.extrinsics
        print(f"  ✓ Extrinsics extracted: shape {extrinsics.shape}")
    else:
        print(f"  ℹ️  Extrinsics not provided by DA3")

    # 清理 GPU 缓存
    if 'cuda' in str(device_obj):
        torch.cuda.empty_cache()

    return depth_maps, intrinsics, fov_values, extrinsics


def depth_fusion_lsq(depth_crafter, depth_da3):
    """修正后的 LSQ: 将 DepthCrafter (Source) 对齐到 DA3 (Target)"""
    original_shape = depth_crafter.shape

    # 扁平化
    inv_Target = 1.0 / depth_da3.flatten()
    inv_Source = 1.0 / depth_crafter.flatten()

    # 构建问题 inv_Target = s * inv_Source + b
    A = np.vstack([inv_Source, np.ones(len(inv_Source))]).T
    y = inv_Target

    # 求解
    x, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    s, b = x[0], x[1]

    print(f"  LSQ Calibration: s={s:.6f}, b={b:.6f}")
    if len(residuals) > 0:
        print(f"  Residuals: {residuals[0]:.6e}")

    # 应用变换到 DepthCrafter 上
    inv_DC_flat = 1.0 / depth_crafter.flatten()
    inv_DC_calibrated = s * inv_DC_flat + b

    # 限制范围 (防止负数或无穷大)
    max_inv = np.max(inv_Target) * 2.0
    inv_DC_calibrated = np.clip(inv_DC_calibrated, 1e-4, max_inv)

    # 转回深度
    D_calibrated_flat = 1.0 / inv_DC_calibrated
    D_calibrated = D_calibrated_flat.reshape(original_shape)

    return D_calibrated, s, b


def fuse_depths(depth_dc, depth_da3, intrinsics_dc, intrinsics_da3,
                extrinsics_da3=None, method='lsq', **kwargs):
    """融合两个深度图

    Args:
        depth_dc: DepthCrafter 深度 (N, H, W)
        depth_da3: DA3 深度 (N, H, W)
        intrinsics_dc: DepthCrafter 内参 (3, 3)
        intrinsics_da3: DA3 内参 (3, 3) 或 (N, 3, 3)
        extrinsics_da3: DA3 外参 (N, 4, 4) - 用于几何融合
        method: 融合方法 ('lsq', 'average', 'weighted', 'max', 'geometric')
    """
    print(f"\nFusing depths using method: {method}")

    # 空间对齐
    if depth_dc.shape != depth_da3.shape:
        print(f"  Resizing DA3 depth {depth_da3.shape} -> {depth_dc.shape}")
        depth_da3 = F.interpolate(
            depth_da3.unsqueeze(1),
            size=depth_dc.shape[-2:],
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

    # 时间对齐检查
    if depth_dc.shape[0] != depth_da3.shape[0]:
        min_len = min(depth_dc.shape[0], depth_da3.shape[0])
        print(f"⚠️ Frame mismatch! Truncating to {min_len} frames.")
        depth_dc = depth_dc[:min_len]
        depth_da3 = depth_da3[:min_len]
        if extrinsics_da3 is not None:
            extrinsics_da3 = extrinsics_da3[:min_len]

    if method == 'geometric' and extrinsics_da3 is not None:
        """论文 3.2 节：基于几何投影的深度融合"""
        print(f"  Using geometric fusion (Warped Depth based)")
        N, H, W = depth_dc.shape
        device = depth_dc.device

        fused_list = []
        for i in range(N):
            print(f"    Processing frame {i+1}/{N}...")

            # 获取当前帧的内外参
            K_dc = intrinsics_dc.to(device) if isinstance(intrinsics_dc, torch.Tensor) else torch.from_numpy(intrinsics_dc).float().to(device)
            K_da3 = intrinsics_da3[i].to(device) if intrinsics_da3.ndim == 3 else intrinsics_da3.to(device)
            P_da3 = torch.from_numpy(extrinsics_da3[i]).float().to(device)

            # 使用 DA3 的外参作为目标，DC 的深度作为源
            warped_depth_dc, valid_mask = get_warped_depth_guidance(
                depth_dc[i].cpu().numpy(),
                K_dc,
                torch.eye(4),  # DC 假设为身份矩阵
                K_da3,
                P_da3
            )

            warped_depth_dc = warped_depth_dc.to(device).squeeze(0).squeeze(0)

            # 融合策略：使用有效像素的加权平均
            valid_mask_2d = valid_mask.squeeze(0).squeeze(0).to(device)

            # 在有效区域，使用扭曲深度；在无效区域，使用原始深度
            fused_frame = torch.where(
                valid_mask_2d > 0.5,
                0.7 * warped_depth_dc + 0.3 * depth_da3[i],  # 有效区域：加权融合
                depth_da3[i]  # 无效区域：使用 DA3 深度
            )

            fused_list.append(fused_frame)

        fused = torch.stack(fused_list, dim=0)
        print(f"  ✓ Geometric fusion completed")

    elif method == 'lsq':
        depth_dc_np = depth_dc.cpu().numpy() if isinstance(depth_dc, torch.Tensor) else depth_dc
        depth_da3_np = depth_da3.cpu().numpy() if isinstance(depth_da3, torch.Tensor) else depth_da3

        # 获取校准后的 DepthCrafter
        depth_dc_calibrated_np, s, b = depth_fusion_lsq(depth_dc_np, depth_da3_np)

        # 结果就是校准后的 DC
        fused = torch.from_numpy(depth_dc_calibrated_np).to(depth_dc.device).float()

    elif method == 'average':
        fused = (depth_dc + depth_da3) / 2.0

    elif method == 'weighted':
        w = kwargs.get('weight_dc', 0.5)
        fused = depth_dc * w + depth_da3 * (1 - w)

    else:
        fused = torch.max(depth_dc, depth_da3)

    print(f"  Fused range: [{fused.min():.4f}, {fused.max():.4f}]")
    return fused


def visualize_depth_video(depth_maps, output_path):
    """将深度图可视化并保存为视频"""
    print(f"  Generating depth visualization video...")

    # 确保形状是 (N, H, W)
    if depth_maps.ndim == 4:
        depth_maps = depth_maps.squeeze(1)

    depth_vis = []
    depth_min = depth_maps.min()
    depth_max = depth_maps.max()
    depth_range = depth_max - depth_min + 1e-8

    for frame_idx in range(len(depth_maps)):
        depth = depth_maps[frame_idx]  # (H, W)

        # 归一化到 [0, 255]
        depth_normalized = (depth - depth_min) / depth_range * 255
        depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)

        # 转为彩色热力图
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

        # 转为 RGB
        depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        depth_vis.append(depth_color_rgb)

    depth_vis = np.stack(depth_vis, axis=0)
    imwrite(output_path, depth_vis, fps=30)
    print(f"  ✓ Depth video saved: {output_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='深度融合脚本 (DepthCrafter + DA3)')

    # 基础参数
    parser.add_argument("--input_video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--scene_name", type=str, required=True, help="场景名称（用于组织输出目录结构）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")
    parser.add_argument("--device", type=str, default='cuda:0', help="计算设备")
    parser.add_argument("--num_frames", type=int, default=-1, help="处理帧数 (-1 = 全部)")

    # DA3 深度参数
    parser.add_argument("--da3_model_dir", type=str, required=True,
                       help="DA3 模型目录路径 (包含 config.json 和 model.safetensors)")
    parser.add_argument("--da3_process_res", type=int, default=504,
                       help="DA3 处理分辨率 (default: 504)")
    parser.add_argument("--rayhead", action='store_true', default=False,
                       help="Ray Head (use_ray_pose): derive camera pose from ray head. Generally slower but more accurate. (default: False)")
    parser.add_argument(
        "--da3_ref_view_strategy",
        type=str,
        default="middle",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="DA3 reference view strategy. For video frames, middle is usually more stable than saddle_balanced.",
    )

    # DepthCrafter 参数
    parser.add_argument("--dc_steps", type=int, default=5, help="DepthCrafter 推理步数")
    parser.add_argument("--fusion_method", type=str, default='lsq',
                       choices=['lsq', 'average', 'weighted', 'max', 'geometric'],
                       help="融合方法。geometric 需要外参支持，使用论文 3.2 节的投影方法")

    # 可视化参数
    parser.add_argument("--save_visualization", action='store_true',
                       help="保存深度可视化视频")

    # 帧数参数
    parser.add_argument("--target_n_frames", type=int, default=49,
                       help="目标帧数（默认 49）")
    parser.add_argument("--max_res", type=int, default=320,
                       help="输入帧最大分辨率（默认 320，超出则等比缩放）")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化设备参数
    device_for_fusion = args.device

    # ==================== 设备检查 ====================
    print("\n" + "="*60)
    print("设备检查")
    print("="*60)
    print(f"  指定设备: {device_for_fusion}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 设备数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"           显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # 验证指定的设备存在
    if 'cuda' in device_for_fusion:
        gpu_id = int(device_for_fusion.split(':')[1])
        if gpu_id >= torch.cuda.device_count():
            print(f"  ❌ 错误: GPU {gpu_id} 不存在！可用 GPU 数: {torch.cuda.device_count()}")
            sys.exit(1)
        print(f"  ✓ GPU {gpu_id} 存在")
        # 设置当前 CUDA 设备
        torch.cuda.set_device(gpu_id)
        print(f"  ✓ 已设置当前 CUDA 设备为 GPU {gpu_id}")

    # ==================== 深度融合 ====================
    print("\n" + "="*60)
    print("深度融合 (DepthCrafter + DA3)")
    print("="*60)

    _, total_frames = get_video_info(args.input_video)
    # 使用参数指定的帧数
    process_len = args.target_n_frames
    print(f"  Using {process_len} frames (original total: {total_frames})")

    # 读取帧
    print("\nLoading frames...")
    frames_np, _ = read_video_frames(args.input_video, process_length=process_len, max_res=args.max_res)
    print(f"Loaded {frames_np.shape[0]} frames")

    # DepthCrafter
    print("\nInitializing DepthCrafter...")
    print(f"  使用设备: {device_for_fusion}")

    # 显存检查
    if 'cuda' in device_for_fusion:
        gpu_id = int(device_for_fusion.split(':')[1])
        mem_info = torch.cuda.mem_get_info(device=gpu_id)
        free_mem = mem_info[0] / 1e9
        total_mem = mem_info[1] / 1e9
        used_mem = total_mem - free_mem
        print(f"  显存使用: {used_mem:.1f}/{total_mem:.1f} GB ({100*used_mem/total_mem:.1f}%)")

    # 使用 model CPU offload 来避免 GPU 内存溢出和 fp16 兼容性问题
    # 这与 recon_fixed_2.py 的做法一致
    dc_model = DepthCrafterDemo(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",  # 使用 model offload，GPU 计算速度快
        device=device_for_fusion
    )
    print(f"  ✓ DepthCrafter 已初始化到: {device_for_fusion}")

    # 检查模型实际在的设备
    if hasattr(dc_model, 'pipe'):
        try:
            device_of_model = next(dc_model.pipe.parameters()).device
            print(f"  模型实际设备: {device_of_model}")
        except:
            pass
    depth_dc, intrinsics_dc = extract_depth_from_depthcrafter(
        frames_np, dc_model, device=device_for_fusion, depth_inference_steps=args.dc_steps
    )
    # 确保 depth_dc 是张量并转换设备
    if not isinstance(depth_dc, torch.Tensor):
        depth_dc = torch.from_numpy(depth_dc)

    device_for_fusion = args.device
    try:
        depth_dc = depth_dc.to(args.device)
        print(f"  ✓ DepthCrafter 深度已移到设备: {args.device}")
    except RuntimeError as e:
        print(f"  ⚠️  设备移动失败: {e}")
        print(f"     降级到 CPU 继续处理...")
        depth_dc = depth_dc.cpu()
        device_for_fusion = 'cpu'

    # 释放 DepthCrafter 模型以腾出显存给 DA3
    print("\n释放 DepthCrafter 显存...")
    del dc_model
    if 'cuda' in str(device_for_fusion):
        torch.cuda.empty_cache()
    print("  ✓ DepthCrafter 已释放")

    # DA3 - 使用 device_for_fusion 以确保一致性
    print("\nInitializing DA3...")
    depth_da3, intrinsics_da3, fov_values, extrinsics_da3 = extract_depth_from_da3(
        args.input_video,
        args.da3_model_dir,
        args.da3_process_res,
        device_for_fusion,  # 使用已验证的设备
        use_ray_pose=args.rayhead,
        max_frames=process_len,
        ref_view_strategy=args.da3_ref_view_strategy,
    )

    # 帧数对齐
    if depth_da3.shape[0] > depth_dc.shape[0]:
        depth_da3 = depth_da3[:depth_dc.shape[0]]

    # DA3 张量应该已经在正确的设备上（来自 extract_depth_from_da3）
    print(f"  DepthCrafter 设备: {depth_dc.device}")
    print(f"  DA3 设备: {depth_da3.device}")

    # 确保两个张量在同一设备上
    if depth_dc.device != depth_da3.device:
        print(f"  ⚠️  深度张量设备不一致，同步到 DC 设备")
        depth_da3 = depth_da3.to(depth_dc.device)

    # 融合
    print("\nFusing depths...")
    depth_fused = fuse_depths(
        depth_dc, depth_da3,
        intrinsics_dc=intrinsics_dc,
        intrinsics_da3=intrinsics_da3,
        extrinsics_da3=extrinsics_da3,
        method=args.fusion_method
    )

    # ==================== 重采样到目标帧数 ====================
    target_n_frames = args.target_n_frames
    current_n_frames = depth_fused.shape[0]

    if target_n_frames != -1 and current_n_frames != target_n_frames:
        print(f"\nResampling all depths to {target_n_frames} frames...")

        # 重采样深度数据的函数
        def resample_depth(depth_tensor, target_frames):
            """将深度数据重采样到目标帧数"""
            current_frames = depth_tensor.shape[0]
            frame_indices = np.linspace(0, current_frames - 1, target_frames)
            depth_list = []
            for idx in frame_indices:
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))
                alpha = idx - idx_floor
                if idx_floor == idx_ceil:
                    depth_list.append(depth_tensor[idx_floor])
                else:
                    depth_list.append(
                        (1 - alpha) * depth_tensor[idx_floor] + alpha * depth_tensor[idx_ceil]
                    )
            return torch.stack(depth_list, dim=0)

        # 重采样所有深度
        depth_dc = resample_depth(depth_dc, target_n_frames)
        depth_da3 = resample_depth(depth_da3, target_n_frames)
        depth_fused = resample_depth(depth_fused, target_n_frames)

        # 重采样外参（如果存在）
        if extrinsics_da3 is not None:
            def resample_extrinsics(extrinsics, target_frames):
                """将外参数据重采样到目标帧数"""
                current_frames = extrinsics.shape[0]
                frame_indices = np.linspace(0, current_frames - 1, target_frames)
                extrinsics_list = []
                for idx in frame_indices:
                    idx_floor = int(np.floor(idx))
                    idx_ceil = int(np.ceil(idx))
                    if idx_floor == idx_ceil:
                        extrinsics_list.append(extrinsics[idx_floor])
                    else:
                        # 对于变换矩阵，使用线性插值
                        alpha = idx - idx_floor
                        ext_interp = (1 - alpha) * extrinsics[idx_floor] + alpha * extrinsics[idx_ceil]
                        extrinsics_list.append(ext_interp)
                return np.stack(extrinsics_list, axis=0)

            if extrinsics_da3.shape[0] != target_n_frames:
                extrinsics_da3 = resample_extrinsics(extrinsics_da3, target_n_frames)
                print(f"  ✓ Resampled extrinsics to {target_n_frames} frames")

        # 相应调整 FOV 值
        if len(fov_values) != target_n_frames:
            fov_indices = np.linspace(0, len(fov_values) - 1, target_n_frames)
            fov_list = []
            for idx in fov_indices:
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))
                alpha = idx - idx_floor
                if idx_floor == idx_ceil:
                    fov_list.append(fov_values[idx_floor])
                else:
                    fov_list.append(
                        (1 - alpha) * fov_values[idx_floor] + alpha * fov_values[idx_ceil]
                    )
            fov_values = np.array(fov_list)

        print(f"  ✓ Resampled to {target_n_frames} frames")
        print(f"    depth_dc shape: {depth_dc.shape}")
        print(f"    depth_da3 shape: {depth_da3.shape}")
        print(f"    depth_fused shape: {depth_fused.shape}")
        print(f"    fov_values shape: {fov_values.shape}")
    else:
        print(f"  ✓ Using all {current_n_frames} frames (no resampling)")


    # 诊断信息：打印深度值和内参
    print("\n" + "="*60)
    print("诊断信息：深度值和内参对比")
    print("="*60)
    print(f"DepthCrafter 深度范围: [{depth_dc.min():.4f}, {depth_dc.max():.4f}]")
    print(f"DA3 深度范围: [{depth_da3.min():.4f}, {depth_da3.max():.4f}]")
    print(f"融合后深度范围: [{depth_fused.min():.4f}, {depth_fused.max():.4f}]")
    print(f"\nDepthCrafter 内参 (假内参):")
    print(f"  fx={intrinsics_dc[0,0]:.2f}, fy={intrinsics_dc[1,1]:.2f}")
    print(f"  cx={intrinsics_dc[0,2]:.2f}, cy={intrinsics_dc[1,2]:.2f}")
    print(f"\nDA3 内参:")
    print(f"  fx={intrinsics_da3[0,0]:.2f}, fy={intrinsics_da3[1,1]:.2f}")
    print(f"  cx={intrinsics_da3[0,2]:.2f}, cy={intrinsics_da3[1,2]:.2f}")
    print("="*60)

    # 使用 DA3 的内参而不是 DC 的假内参
    # 如果融合后的深度分辨率与 DA3 不同，需要调整内参
    final_H, final_W = depth_fused.shape[-2:]
    da3_H, da3_W = depth_da3.shape[-2:]

    if final_H != da3_H or final_W != da3_W:
        print(f"\n  Adjusting DA3 intrinsics from ({da3_H}, {da3_W}) to ({final_H}, {final_W})")
        scale_x = final_W / da3_W
        scale_y = final_H / da3_H
        intrinsics_final = intrinsics_da3.clone()
        intrinsics_final[0, 0] *= scale_x  # fx
        intrinsics_final[1, 1] *= scale_y  # fy
        intrinsics_final[0, 2] *= scale_x  # cx
        intrinsics_final[1, 2] *= scale_y  # cy
    else:
        intrinsics_final = intrinsics_da3

    print(f"\n最终使用的内参:")
    print(f"  fx={intrinsics_final[0,0]:.2f}, fy={intrinsics_final[1,1]:.2f}")
    print(f"  cx={intrinsics_final[0,2]:.2f}, cy={intrinsics_final[1,2]:.2f}")

    # 保存融合深度数据和内参
    print("\nSaving depth and intrinsics results...")

    # 创建输出子目录
    dc_depth_dir = os.path.join(args.output_dir, "depthcrafter", args.scene_name)
    da3_depth_dir = os.path.join(args.output_dir, "da3", args.scene_name)
    fused_depth_dir = os.path.join(args.output_dir, "fusion", args.scene_name)

    os.makedirs(dc_depth_dir, exist_ok=True)
    os.makedirs(da3_depth_dir, exist_ok=True)
    os.makedirs(fused_depth_dir, exist_ok=True)

    # FOV 值处理
    if fov_values is None:
        print(f"  ⚠️  FOV not provided, using default value 55.0 for all frames")
        fov_values = np.full(depth_da3.shape[0], 55.0)
    else:
        # 确保 fov_values 是 numpy 数组
        if not isinstance(fov_values, np.ndarray):
            fov_values = np.array(fov_values)

        # 处理 NaN 值
        if np.isnan(fov_values).any():
            print(f"  ⚠️  Found {np.isnan(fov_values).sum()} NaN FOV values, replacing with default 55.0")
            fov_values = np.nan_to_num(fov_values, nan=55.0)

        # 如果 fov_values 是 1D 数组但长度不匹配
        if fov_values.ndim == 0 or (fov_values.ndim == 1 and len(fov_values) == 1):
            fov_val = float(fov_values.item() if hasattr(fov_values, 'item') else fov_values)
            print(f"  Single FOV value: {fov_val}, replicating for all frames")
            fov_values = np.full(depth_da3.shape[0], fov_val)
        elif len(fov_values) != depth_da3.shape[0]:
            print(f"  ⚠️  FOV length {len(fov_values)} != depth frames {depth_da3.shape[0]}")
            fov_values = fov_values[:depth_da3.shape[0]]

    print(f"  Final FOV values: shape={fov_values.shape}, min={np.min(fov_values):.2f}, max={np.max(fov_values):.2f}")

    # 分别保存每一帧的深度数据（类似 run_mono-depth_demo.sh 的方式）
    # DepthCrafter: 保存为 .npy 格式（2D: H x W）
    print(f"  Saving DepthCrafter depth to: {dc_depth_dir}")
    print(f"    depth_dc shape before saving: {depth_dc.shape}")
    for i in range(depth_dc.shape[0]):
        depth_frame = depth_dc[i].cpu().numpy()
        # 确保是 2D
        if depth_frame.ndim > 2:
            print(f"    ⚠️  Frame {i} has {depth_frame.ndim}D shape {depth_frame.shape}, squeezing to 2D")
            depth_frame = depth_frame.squeeze()
        np.save(os.path.join(dc_depth_dir, f"frame_{i:05d}.npy"), depth_frame)
    print(f"  ✓ Saved {depth_dc.shape[0]} DepthCrafter frames")

    # DA3: 保存为 .npz 格式（包含 depth(2D: H x W) 和 fov，兼容 test_demo.py）
    print(f"  Saving DA3 depth to: {da3_depth_dir}")
    print(f"    depth_da3 shape before saving: {depth_da3.shape}")
    for i in range(depth_da3.shape[0]):
        depth_frame = depth_da3[i].cpu().numpy()
        # 确保是 2D
        if depth_frame.ndim > 2:
            print(f"    ⚠️  Frame {i} has {depth_frame.ndim}D shape {depth_frame.shape}, squeezing to 2D")
            depth_frame = depth_frame.squeeze()
        # 确保 fov_val 是标量
        fov_val = float(fov_values[i]) if isinstance(fov_values, np.ndarray) else float(fov_values)
        np.savez(
            os.path.join(da3_depth_dir, f"frame_{i:05d}.npz"),
            depth=depth_frame,  # 2D
            fov=np.array(fov_val)
        )
    print(f"  ✓ Saved {depth_da3.shape[0]} DA3 frames (with FOV)")

    # 融合深度: 保存为 .npz 格式（包含 depth(2D: H x W) 和 fov）
    print(f"  Saving fused depth to: {fused_depth_dir}")
    print(f"    depth_fused shape before saving: {depth_fused.shape}")
    for i in range(depth_fused.shape[0]):
        depth_frame = depth_fused[i].cpu().numpy()
        # 确保是 2D
        if depth_frame.ndim > 2:
            print(f"    ⚠️  Frame {i} has {depth_frame.ndim}D shape {depth_frame.shape}, squeezing to 2D")
            depth_frame = depth_frame.squeeze()
        # 确保 fov_val 是标量
        fov_val = float(fov_values[i]) if isinstance(fov_values, np.ndarray) else float(fov_values)
        np.savez(
            os.path.join(fused_depth_dir, f"frame_{i:05d}.npz"),
            depth=depth_frame,  # 2D
            fov=np.array(fov_val)
        )
    print(f"  ✓ Saved {depth_fused.shape[0]} fused frames (with FOV)")

    # 单独保存 DA3 内参（放在输出目录根目录）
    da3_intrinsics_path = os.path.join(args.output_dir, "da3_intrinsics.npy")
    np.save(da3_intrinsics_path, intrinsics_final.cpu().numpy())
    print(f"  ✓ DA3 intrinsics saved: {da3_intrinsics_path}")

    # 单独保存 DA3 外参 (extrinsics) - 相机位姿
    if extrinsics_da3 is not None:
        da3_extrinsics_path = os.path.join(args.output_dir, "da3_extrinsics.npy")
        np.save(da3_extrinsics_path, extrinsics_da3)
        print(f"  ✓ DA3 extrinsics saved: {da3_extrinsics_path}")
        print(f"    Shape: {extrinsics_da3.shape} (N={extrinsics_da3.shape[0]} frames, 4x4 matrices)")
        print(f"    Format: Each frame has a 4x4 camera-to-world (or world-to-camera) transformation matrix")
    else:
        print(f"  ⚠️  DA3 extrinsics not available, skipping extrinsics save")

    # 可视化深度
    if args.save_visualization:
        print("\n" + "="*60)
        print("生成深度可视化视频")
        print("="*60)
        vis_dc_path = os.path.join(args.output_dir, "depthcrafter_depth_vis.mp4")
        vis_da3_path = os.path.join(args.output_dir, "da3_depth_vis.mp4")
        vis_fused_path = os.path.join(args.output_dir, "fused_depth_vis.mp4")

        visualize_depth_video(depth_dc.cpu().numpy(), vis_dc_path)
        visualize_depth_video(depth_da3.cpu().numpy(), vis_da3_path)
        visualize_depth_video(depth_fused.cpu().numpy(), vis_fused_path)

    # ==================== 完成 ====================
    print("\n" + "="*60)
    print("✅ 深度融合完成!")
    print("="*60)
    print(f"📁 输出目录: {args.output_dir}")
    print(f"\n📊 生成的目录结构 (兼容 test_demo.py):")
    print(f"   {args.output_dir}/")
    print(f"   ├── depthcrafter/{args.scene_name}/")
    print(f"   │   ├── frame_00001.npy")
    print(f"   │   ├── frame_00002.npy")
    print(f"   │   └── ...")
    print(f"   ├── da3/{args.scene_name}/")
    print(f"   │   ├── frame_00001.npz  (包含 depth + fov)")
    print(f"   │   ├── frame_00002.npz  (包含 depth + fov)")
    print(f"   │   └── ...")
    print(f"   ├── fusion/{args.scene_name}/")
    print(f"   │   ├── frame_00001.npz  (包含 depth + fov)")
    print(f"   │   ├── frame_00002.npz  (包含 depth + fov)")
    print(f"   │   └── ...")
    print(f"   ├── da3_intrinsics.npy (Nx3x3 内参矩阵)")
    if extrinsics_da3 is not None:
        print(f"   └── da3_extrinsics.npy (Nx4x4 外参矩阵 - 相机位姿)")
    print(f"\n🔧 用于 test_demo.py 的配置示例:")
    print(f"   --mono_depth_path {args.output_dir}/depthcrafter")
    print(f"   --metric_depth_path {args.output_dir}/da3")

    print(f"\n📋 深度融合方法说明:")
    if args.fusion_method == 'geometric':
        print(f"   • 当前使用: Geometric Fusion (论文 3.2 节)")
        print(f"   • 原理: 利用内外参进行投影空间的深度融合")
        print(f"   • 流程: 反投影 → 3D变换 → 重投影 → 扭曲深度采样")
        print(f"   • 优点: 物理约束完整，支持复杂相机运动")
        print(f"   • 建议: 启用 --rayhead 以获得更准确的外参估计")
    else:
        print(f"   • 当前使用: {args.fusion_method.upper()} Fusion")
        print(f"   • Geometric Fusion 可用: python depth_fusion_da3.py ... --fusion_method geometric")

    if args.save_visualization:
        print(f"\n   可视化视频:")
        print(f"   ├── depthcrafter_depth_vis.mp4")
        print(f"   ├── da3_depth_vis.mp4")
        print(f"   └── fused_depth_vis.mp4")


if __name__ == '__main__':
    main()
