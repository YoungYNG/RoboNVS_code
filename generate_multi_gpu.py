import argparse
import os
import torch
import torchvision
from diffsynth import ModelManager, WanVideoPipeline, save_video, load_state_dict
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
import numpy as np
from diffsynth.models.camera import CamVidEncoder
import cv2
from accelerate import infer_auto_device_map, dispatch_model


def get_video_resolution(video_path):
    """Get the resolution of a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height, width


def load_video_frames(video_path, num_frames, height, width):
    """Load and process video frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        print(f"Warning: Video has only {frame_count} frames, but {num_frames} required")
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    frames = np.array(frames)
    video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return video_tensor
def load_mask_frames(mask_path, num_frames, height, width):
    """Load and process mask frames"""
    cap = cv2.VideoCapture(mask_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        print(f"Warning: Mask video has only {frame_count} frames, but {num_frames} required")
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            import random
            kernel = random.randint(2, 8)
            frame = cv2.erode(frame, np.ones((kernel, kernel), np.uint8), iterations=1)
            frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    frames = np.array(frames)
    mask_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    mask_tensor = (mask_tensor / 255.0 > 0.5).float()
    return mask_tensor
def add_lora_to_model(model, lora_rank=16, lora_alpha=16.0, lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                     init_lora_weights="kaiming", pretrained_path=None, state_dict_converter=None):
    """Add LoRA to model"""
    if init_lora_weights == "kaiming":
        init_lora_weights = True
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    if pretrained_path is not None:
        state_dict = load_state_dict(pretrained_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(f"LORA: {num_updated_keys} parameters loaded from {pretrained_path}. {num_unexpected_keys} unexpected.")


def run_inference_multigpu(pipe, prompt, negative_prompt, video, mask,
                           num_inference_steps, seed, cfg_scale,
                           tiled, tile_size, tile_stride,
                           height, width, num_frames,
                           dit_device, vae_device, text_device):
    """
    Custom inference function for multi-GPU setup.
    This bypasses the pipeline's automatic device management.
    """
    from tqdm import tqdm
    from einops import rearrange

    # Check parameters
    if num_frames % 4 != 1:
        num_frames = (num_frames + 2) // 4 * 4 + 1
        print(f"Adjusted num_frames to {num_frames}")

    tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    # Set timesteps on scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)

    # Generate initial noise
    print(f"Generating noise on {dit_device}...")
    noise = pipe.generate_noise(
        (1, 16, (num_frames - 1) // 4 + 1, height//8, width//8),
        seed=seed, device="cpu", dtype=torch.float32
    ).to(dit_device)
    latents = noise

    # For distributed DiT, ensure latents are on the first device
    if hasattr(pipe, 'dit_is_distributed') and pipe.dit_is_distributed:
        # Get the device of the first layer of DiT
        first_device = next(pipe.dit.parameters()).device
        latents = latents.to(first_device)
        print(f"  -> Latents moved to DiT's first device: {first_device}")

    # Encode prompts (on text_device)
    print(f"Encoding prompts...")
    # Temporarily override pipe.device for text encoding
    original_device = pipe.device
    pipe.device = text_device

    with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
        prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
        # Context can be a tensor or list - handle both
        if "context" in prompt_emb_posi:
            if isinstance(prompt_emb_posi["context"], list):
                prompt_emb_posi["context"] = [c.to(text_device) if isinstance(c, torch.Tensor) else c
                                              for c in prompt_emb_posi["context"]]
            elif isinstance(prompt_emb_posi["context"], torch.Tensor):
                prompt_emb_posi["context"] = prompt_emb_posi["context"].to(text_device)

        if cfg_scale != 1.0:
            prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)
            if "context" in prompt_emb_nega:
                if isinstance(prompt_emb_nega["context"], list):
                    prompt_emb_nega["context"] = [c.to(text_device) if isinstance(c, torch.Tensor) else c
                                                  for c in prompt_emb_nega["context"]]
                elif isinstance(prompt_emb_nega["context"], torch.Tensor):
                    prompt_emb_nega["context"] = prompt_emb_nega["context"].to(text_device)

    # Restore original device
    pipe.device = original_device

    # Encode image (uses image_encoder, vae, and text_device)
    print(f"Encoding image features...")
    # Get actual device for image encoder
    image_encoder_device = next(pipe.image_encoder.parameters()).device if pipe.image_encoder is not None else vae_device

    with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
        if video is not None and pipe.image_encoder is not None:
            # Manually encode instead of using pipe.encode_images to control devices properly
            batch_size = video.shape[0]
            clip_context = []
            y = []

            for i in range(batch_size):
                # Convert to PIL image
                import torchvision.transforms.functional as TF
                image_tensor = video[i, :, 0]  # Get first frame [C, H, W]
                image_pil = TF.to_pil_image(image_tensor.float() * 0.5 + 0.5)

                # Resize and preprocess
                image_pil = image_pil.resize((width, height))
                image = pipe.preprocess_image(image_pil).to(image_encoder_device)

                # Encode with image encoder (on image_encoder_device)
                clip_ctx = pipe.image_encoder.encode_image([image])
                clip_context.append(clip_ctx)

                # Prepare mask for VAE
                msk = torch.ones(1, num_frames, height//8, width//8, device=vae_device)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
                msk = msk.transpose(1, 2)[0]

                # Encode with VAE (on vae_device)
                old_pipe_device = pipe.device
                pipe.device = vae_device

                image_for_vae = image.to(vae_device)
                vae_input = torch.concat([image_for_vae.transpose(0, 1),
                                         torch.zeros(3, num_frames-1, height, width).to(vae_device)], dim=1)
                y_latent = pipe.vae.encode([vae_input], device=vae_device)[0]
                y_latent = torch.concat([msk, y_latent])
                y.append(y_latent)

                pipe.device = old_pipe_device

            clip_context = torch.cat(clip_context, dim=0)
            y = torch.stack(y, dim=0)

            image_emb = {"clip_fea": clip_context.to(text_device), "y": y.to(vae_device)}
        else:
            image_emb = {}

        # Encode camera rays (uses vae_device and dit_device for camera_encoder)
        pipe.device = vae_device
        video_vae = video.to(vae_device)
        mask_vae = mask.to(vae_device)
        ray_latent = pipe.encode_rays(video_vae, mask_vae)
        image_emb["ray_latent"] = ray_latent.to(dit_device)

        # Restore device to dit_device
        pipe.device = dit_device

    # Prepare extra input
    extra_input = pipe.prepare_extra_input(latents)

    # Helper function to move data to device
    def move_to_device(data, device):
        """Move tensor/list/dict to device"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        elif isinstance(data, dict):
            return {k: move_to_device(v, device) for k, v in data.items()}
        else:
            return data

    # Denoise loop
    print(f"Denoising...")
    if hasattr(pipe, 'dit_is_distributed') and pipe.dit_is_distributed:
        # Get first device for distributed DiT
        first_device = next(pipe.dit.parameters()).device
    else:
        first_device = dit_device

    with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="Denoising")):
            # For distributed DiT, move all inputs to first device
            # For non-distributed, we need to move to dit_device
            if hasattr(pipe, 'dit_is_distributed') and pipe.dit_is_distributed:
                # Distributed DiT: move all inputs to first device (accelerate will handle the rest)
                timestep_input = timestep.unsqueeze(0).to(dtype=torch.float32, device=first_device)
                prompt_posi_dit = move_to_device(prompt_emb_posi, first_device)
                image_emb_dit = move_to_device(image_emb, first_device)
                if cfg_scale != 1.0:
                    prompt_nega_dit = move_to_device(prompt_emb_nega, first_device)
            else:
                # Non-distributed DiT: move all inputs to dit_device
                timestep_input = timestep.unsqueeze(0).to(dtype=torch.float32, device=dit_device)
                prompt_posi_dit = move_to_device(prompt_emb_posi, dit_device)
                image_emb_dit = {}
                if "clip_fea" in image_emb:
                    image_emb_dit["clip_fea"] = image_emb["clip_fea"].to(dit_device)
                if "y" in image_emb:
                    image_emb_dit["y"] = image_emb["y"].to(dit_device)
                if "ray_latent" in image_emb:
                    image_emb_dit["ray_latent"] = image_emb["ray_latent"]
                if cfg_scale != 1.0:
                    prompt_nega_dit = move_to_device(prompt_emb_nega, dit_device)

            # Forward pass
            noise_pred_posi = pipe.dit(latents, timestep=timestep_input, **prompt_posi_dit, **image_emb_dit, **extra_input)

            if cfg_scale != 1.0:
                noise_pred_nega = pipe.dit(latents, timestep=timestep_input, **prompt_nega_dit, **image_emb_dit, **extra_input)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

            # Ensure latents stay on the correct device
            if hasattr(pipe, 'dit_is_distributed') and pipe.dit_is_distributed:
                # For distributed DiT, keep latents on the first device
                if latents.device != first_device:
                    latents = latents.to(first_device)

    # Decode (on vae_device)
    print(f"Decoding video...")
    latents_vae = latents.to(vae_device)

    # Temporarily set pipe.device to vae_device for decoding
    original_device = pipe.device
    pipe.device = vae_device

    with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
        frames = pipe.decode_video(latents_vae, **tiler_kwargs)

    # Restore original device
    pipe.device = original_device

    # Convert to output format
    frames = pipe.tensor2video(frames[0])
    return frames


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Video Inference with RoboNVS (Multi-GPU)')
    parser.add_argument('--color_video', type=str, required=True, help='Path to condition video')
    parser.add_argument('--mask_video', type=str, required=True, help='Path to mask video')
    parser.add_argument('--output_video', type=str, required=True, help='Path to output video')
    parser.add_argument('--height', type=int, default=None, help='Output height (default: auto-detect from input video)')
    parser.add_argument('--width', type=int, default=None, help='Output width (default: auto-detect from input video)')
    parser.add_argument('--num_frames', type=int, default=49, help='Number of frames to process')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
    # Model paths
    parser.add_argument('--ex4d_path', type=str, default='models/RoboNVS/RoboNVS_lora16.pt', help='Path to LoRA checkpoint for RoboNVS')
    parser.add_argument('--text_encoder_path', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth')
    parser.add_argument('--vae_path', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth')
    parser.add_argument('--clip_path', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')
    parser.add_argument('--dit_dir', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/')
    # Multi-GPU settings
    parser.add_argument('--device', type=str, default=None, help='Primary device to use (e.g., cuda:3)')
    parser.add_argument('--gpu_ids', type=str, default='2,3,5,7', help='Comma-separated GPU IDs to use')
    parser.add_argument('--max_memory_per_gpu', type=int, default=18,
                       help='Max memory (GB) per GPU for DiT distribution (lower=more even distribution, default: 18)')
    parser.add_argument('--exclude_from_vae_text', type=str, default='',
                       help='GPU IDs to exclude from VAE/Text placement (e.g., "2" to keep GPU 2 lighter)')
    # Memory optimization
    parser.add_argument('--enable_tiled', type=str2bool, nargs='?', const=True, default=True,
                       help='Enable tiled rendering to save memory (default: True). Use --enable_tiled=False to disable.')
    parser.add_argument('--tile_size_h', type=int, default=20,
                       help='Tile height for VAE encoding/decoding (smaller=less memory, default: 20)')
    parser.add_argument('--tile_size_w', type=int, default=30,
                       help='Tile width for VAE encoding/decoding (smaller=less memory, default: 30)')
    parser.add_argument('--tile_stride_h', type=int, default=10,
                       help='Tile stride height (default: 10)')
    parser.add_argument('--tile_stride_w', type=int, default=15,
                       help='Tile stride width (default: 15)')
    parser.add_argument('--vram_management', type=float, default=0.1,
                       help='VRAM management parameter (0.0-1.0, lower=less memory, default: 0.1)')
    parser.add_argument('--cpu_offload', action='store_true', default=False,
                       help='Enable CPU offloading to save GPU memory (slower but much less VRAM)')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                       help='Classifier-free guidance scale (lower=less memory, default: 1.0)')
    args = parser.parse_args()

    # Parse GPU IDs - prioritize --device if provided
    if args.device:
        # Extract GPU ID from device string (e.g., "cuda:3" -> 3)
        if ':' in args.device:
            primary_gpu = int(args.device.split(':')[1])
        else:
            primary_gpu = 0
        gpu_ids = [primary_gpu]
        use_single_gpu = True
        print(f"Using single device: {args.device}")
    else:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        use_single_gpu = len(gpu_ids) == 1
        if use_single_gpu:
            print(f"Using single GPU: cuda:{gpu_ids[0]}")
        else:
            print(f"Using multiple GPUs: {gpu_ids}")

    dit_device = f"cuda:{gpu_ids[0]}"
    vae_device = f"cuda:{gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0]}"
    text_device = f"cuda:{gpu_ids[2] if len(gpu_ids) > 2 else gpu_ids[0]}"
    print(f"Device mapping: DiT={dit_device}, VAE={vae_device}, Text={text_device}")

    # Validate input files
    if not os.path.exists(args.color_video):
        raise FileNotFoundError(f"Color video not found: {args.color_video}")
    if not os.path.exists(args.mask_video):
        raise FileNotFoundError(f"Mask video not found: {args.mask_video}")

    # Auto-detect resolution if not specified
    if args.height is None or args.width is None:
        print("Auto-detecting video resolution from input...")
        detected_height, detected_width = get_video_resolution(args.color_video)
        if args.height is None:
            args.height = detected_height
            print(f"  Using detected height: {args.height}")
        if args.width is None:
            args.width = detected_width
            print(f"  Using detected width: {args.width}")
    else:
        print(f"Using specified resolution: {args.width}x{args.height}")

    # Create output directory
    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Load videos
    print(f"Loading mask video: {args.mask_video}")
    mask_tensor = load_mask_frames(args.mask_video, args.num_frames, args.height, args.width)
    print(f"Loading color video: {args.color_video}")
    color_tensor = load_video_frames(args.color_video, args.num_frames, args.height, args.width)
    color_tensor = (color_tensor * mask_tensor).to(torch.bfloat16) * 2 - 1
    mask_tensor = mask_tensor.to(torch.bfloat16) * 2 - 1

    # Clear GPU cache before loading models
    torch.cuda.empty_cache()

    dit_paths = [
        os.path.join(args.dit_dir, f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors")
        for i in range(1, 8)
    ]

    if use_single_gpu:
        # Single GPU mode: Load to CPU first, then use VRAM management
        print("Single GPU mode: Loading models to CPU first (to avoid GPU OOM during model loading)...")
        print(f"Loading DiT model to CPU...")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([dit_paths])
        print(f"Loading VAE to CPU...")
        vae_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        vae_manager.load_models([args.vae_path])
        # Merge models from vae_manager into model_manager
        model_manager.model.extend(vae_manager.model)
        model_manager.model_path.extend(vae_manager.model_path)
        model_manager.model_name.extend(vae_manager.model_name)
        print(f"Loading text encoders to CPU...")
        text_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        text_manager.load_models([args.text_encoder_path, args.clip_path])
        # Merge models from text_manager into model_manager
        model_manager.model.extend(text_manager.model)
        model_manager.model_path.extend(text_manager.model_path)
        model_manager.model_name.extend(text_manager.model_name)

        # Create pipeline with primary device
        print(f"Creating pipeline with device={dit_device}...")
        pipe = WanVideoPipeline.from_model_manager(model_manager, device=dit_device)
    else:
        # Multi-GPU mode: Load models to different GPUs
        print("Multi-GPU mode: Distributing models across GPUs...")

        # Load all models to CPU first to avoid conflicts
        print(f"Step 1: Loading all models to CPU...")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([dit_paths, args.text_encoder_path, args.vae_path, args.clip_path])

        # Create pipeline on CPU
        print(f"Step 2: Creating pipeline...")
        pipe = WanVideoPipeline.from_model_manager(model_manager, device="cpu")

        # Step 3: Distribute DiT model across multiple GPUs using model parallelism
        print(f"Step 3: Using model parallelism for DiT (14B params)...")
        print(f"  -> Splitting DiT across ALL GPUs {gpu_ids}...")

        # Use ALL GPUs for DiT to maximize distribution
        dit_gpus = gpu_ids
        # Reduce memory quota to force more aggressive distribution across GPUs
        # This will make GPU 2 less loaded by distributing more layers to other GPUs
        max_memory = {i: f"{args.max_memory_per_gpu}GB" for i in dit_gpus}
        max_memory["cpu"] = "100GB"

        print(f"  -> Using aggressive distribution strategy ({args.max_memory_per_gpu}GB limit per GPU)")
        print(f"  -> Lower limit = more even distribution (try --max_memory_per_gpu 16 for even more even distribution)")

        # Use accelerate to automatically split DiT across GPUs
        # Don't split individual transformer blocks or attention modules
        no_split_module_classes = ["WanAttentionBlock", "WanSelfAttention", "WanCrossAttention"]

        # Try automatic distribution first with very low memory limit to force spreading
        try:
            device_map = infer_auto_device_map(
                pipe.dit,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=torch.bfloat16
            )

            # Check if all GPUs are used
            used_gpus = set(d for d in device_map.values() if isinstance(d, int))
            unused_gpus = set(dit_gpus) - used_gpus

            if unused_gpus:
                print(f"  -> Warning: GPUs {unused_gpus} not used by auto distribution")
                print(f"  -> Attempting manual balanced distribution...")

                # Manual distribution: Split 40 blocks evenly across all GPUs
                num_gpus = len(dit_gpus)
                blocks_per_gpu = 40 // num_gpus
                remainder = 40 % num_gpus

                device_map = {}
                # Embeddings on first GPU
                device_map['patch_embedding'] = dit_gpus[0]
                device_map['text_embedding'] = dit_gpus[0]
                device_map['time_embedding'] = dit_gpus[0]
                device_map['time_projection'] = dit_gpus[0]

                # Distribute blocks
                block_idx = 0
                for gpu_idx, gpu_id in enumerate(dit_gpus):
                    # Give extra blocks to first GPUs if there's remainder
                    num_blocks_this_gpu = blocks_per_gpu + (1 if gpu_idx < remainder else 0)
                    for _ in range(num_blocks_this_gpu):
                        if block_idx < 40:
                            device_map[f'blocks.{block_idx}'] = gpu_id
                            block_idx += 1

                # Head and img_emb on last GPU
                device_map['head'] = dit_gpus[-1]
                device_map['img_emb'] = dit_gpus[-1]

                print(f"  -> Manual distribution: {blocks_per_gpu} blocks per GPU (with {remainder} GPUs getting +1)")
        except Exception as e:
            print(f"  -> Error in device mapping: {e}")
            raise

        # Count modules per device
        device_counts = {}
        for key, device in device_map.items():
            if device not in device_counts:
                device_counts[device] = []
            device_counts[device].append(key)

        pipe.dit = dispatch_model(pipe.dit, device_map=device_map)

        # Smart device assignment: Put VAE and Text Encoder on GPUs with less DiT load
        # Count how many DiT layers are on each GPU
        dit_layer_counts = {}
        for key, device in device_map.items():
            if isinstance(device, int):
                dit_layer_counts[device] = dit_layer_counts.get(device, 0) + 1

        # Parse excluded GPUs for VAE/Text placement
        excluded_gpus = set()
        if args.exclude_from_vae_text:
            excluded_gpus = set(int(x.strip()) for x in args.exclude_from_vae_text.split(',') if x.strip())
            print(f"\n  -> Excluding GPU(s) {excluded_gpus} from VAE/Text placement")

        # Check available GPU memory before placing models
        gpu_free_memory = {}
        for gpu_id in dit_gpus:
            if gpu_id not in excluded_gpus:
                torch.cuda.empty_cache()
                free_mem = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                free_mem_gb = free_mem / 1024**3
                gpu_free_memory[gpu_id] = free_mem_gb

                # Exclude GPUs with insufficient memory for Text Encoder (needs ~10GB)
                if free_mem_gb < 11.0:  # Need at least 11GB for Text Encoder
                    excluded_gpus.add(gpu_id)

        # Find GPU with least DiT load for VAE, excluding specified GPUs
        # Sort by layer count (ascending), but filter out excluded GPUs
        sorted_gpus = sorted(dit_layer_counts.items(), key=lambda x: x[1])
        available_gpus_for_vae = [(gpu, count) for gpu, count in sorted_gpus if gpu not in excluded_gpus]

        if not available_gpus_for_vae:
            raise RuntimeError("No GPUs available for VAE/Text placement! All GPUs are excluded or have insufficient memory.")

        if len(available_gpus_for_vae) >= 4:
            # Text Encoder is biggest (~10GB), VAE is medium (~4GB), Image is small (~1GB)
            # Put Text on lightest, VAE on second, Image on third
            text_target_gpu = f"cuda:{available_gpus_for_vae[0][0]}"     # Lightest (Text is biggest ~10GB)
            vae_target_gpu = f"cuda:{available_gpus_for_vae[1][0]}"      # Second lightest (VAE ~4GB)
            image_target_gpu = f"cuda:{available_gpus_for_vae[2][0]}"    # Third lightest (Image ~1GB)
        elif len(available_gpus_for_vae) >= 3:
            # Put Text, VAE, and Image on 3 different GPUs with least DiT load
            text_target_gpu = f"cuda:{available_gpus_for_vae[0][0]}"
            vae_target_gpu = f"cuda:{available_gpus_for_vae[1][0]}"
            image_target_gpu = f"cuda:{available_gpus_for_vae[2][0]}"
        elif len(available_gpus_for_vae) == 2:
            vae_target_gpu = f"cuda:{available_gpus_for_vae[0][0]}"
            text_target_gpu = f"cuda:{available_gpus_for_vae[1][0]}"
            image_target_gpu = text_target_gpu  # Share with text
        elif len(available_gpus_for_vae) == 1:
            vae_target_gpu = f"cuda:{available_gpus_for_vae[0][0]}"
            text_target_gpu = vae_target_gpu
            image_target_gpu = vae_target_gpu
        else:
            # Fallback if all GPUs are excluded
            vae_target_gpu = f"cuda:{sorted_gpus[0][0]}" if len(sorted_gpus) > 0 else vae_device
            text_target_gpu = f"cuda:{sorted_gpus[1][0]}" if len(sorted_gpus) > 1 else text_device
            image_target_gpu = f"cuda:{sorted_gpus[2][0]}" if len(sorted_gpus) > 2 else text_device
            print(f"  -> Warning: All GPUs excluded, using fallback assignment")

        # Move VAE to GPU with lighter load
        pipe.vae = pipe.vae.to(vae_target_gpu)

        # Move Text Encoder to another GPU with lighter load
        pipe.text_encoder = pipe.text_encoder.to(text_target_gpu)

        if pipe.image_encoder is not None:
            pipe.image_encoder = pipe.image_encoder.to(image_target_gpu)

        # Store device mappings in pipe for later use
        pipe.dit_device = dit_device
        pipe.vae_device = vae_target_gpu
        pipe.text_device = text_target_gpu
        pipe.image_device = image_target_gpu
        pipe.device = dit_device  # Primary device
        pipe.dit_is_distributed = True  # Flag to indicate DiT is distributed

        print(f"\nMulti-GPU setup: DiT across {dit_gpus}, VAE={vae_target_gpu}, Text={text_target_gpu}, Image={image_target_gpu}")

    # Create camera encoder (will be moved to appropriate device later)
    pipe.camera_encoder = CamVidEncoder(16, 1024, 5120).to("cpu", dtype=torch.bfloat16)

    # Enable VRAM management only for single GPU mode
    if use_single_gpu:
        if args.cpu_offload:
            pipe.enable_vram_management(num_persistent_param_in_dit=None)  # Maximum offloading
        else:
            # Use much smaller value for 24GB GPUs
            num_persistent = int(args.vram_management * 1e9) if args.vram_management < 1 else None
            pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent)

        # Move camera encoder after VRAM management
        pipe.camera_encoder = pipe.camera_encoder.to(dit_device, dtype=torch.bfloat16)
        pipe.camera_encoder.eval()
    else:
        # In multi-GPU mode, we don't use VRAM management
        # Move camera encoder to VAE device since it processes VAE outputs
        camera_target_device = pipe.vae_device if hasattr(pipe, 'vae_device') else vae_device
        pipe.camera_encoder = pipe.camera_encoder.to(camera_target_device, dtype=torch.bfloat16)
        pipe.camera_encoder.eval()

        # Override the load_models_to_device method to prevent automatic device movement
        def dummy_load_models_to_device(model_names):
            """Dummy function to prevent automatic model movement in multi-GPU mode"""
            pass

        pipe.load_models_to_device = dummy_load_models_to_device

    # Add LoRA (after VRAM management is enabled)
    add_lora_to_model(pipe.denoising_model(), pretrained_path=args.ex4d_path, lora_rank=16, lora_alpha=16.0)
    pipe.load_cam(args.ex4d_path)

    # Aggressively clean up GPU memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Set prompts
    prompt = "Inpaint a high quality robot manipulation scene with a realistic robot arm, smooth motion, realistic lighting, and detailed texture.Fill in the missing parts naturally and coherently to complete the scene."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEGcompression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messybackground, three legs, many people in the background, walking backwards"
    # Clear GPU cache before inference
    torch.cuda.empty_cache()

    # Prepare video tensors
    input_cond_cpu = color_tensor[None]
    input_mask_cpu = mask_tensor[None]

    # Run inference
    print("\nRunning inference...")
    print(f"Tiled rendering: {args.enable_tiled}")

    input_cond = input_cond_cpu.to(dit_device)
    input_mask = input_mask_cpu.to(dit_device)

    with torch.no_grad():
        if use_single_gpu:
            # Single GPU mode: Use standard pipeline
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                output_video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    video=input_cond,
                    mask=input_mask,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    cfg_scale=args.cfg_scale,
                    tiled=args.enable_tiled,
                    tile_size=(args.tile_size_h, args.tile_size_w),
                    tile_stride=(args.tile_stride_h, args.tile_stride_w),
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                )
        else:
            # Multi-GPU mode: Use custom inference function
            # Use the updated device mappings from pipe
            actual_vae_device = pipe.vae_device if hasattr(pipe, 'vae_device') else vae_device
            actual_text_device = pipe.text_device if hasattr(pipe, 'text_device') else text_device

            output_video = run_inference_multigpu(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                video=input_cond,
                mask=input_mask,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                tiled=args.enable_tiled,
                tile_size=(args.tile_size_h, args.tile_size_w),
                tile_stride=(args.tile_stride_h, args.tile_stride_w),
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                dit_device=dit_device,
                vae_device=actual_vae_device,
                text_device=actual_text_device,
            )

    # Save output video
    print(f"Saving output video: {args.output_video}")
    save_video(output_video, args.output_video, fps=15, quality=8)
    print("Done!")
if __name__ == "__main__":
    main()
