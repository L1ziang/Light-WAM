import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lightwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from lightwam.datasets.lerobot.processors.lightwam_processor import LightWAMProcessor
from lightwam.utils.pytorch_utils import get_default_device
from lightwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from lightwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from lightwam.utils.config_compat import load_compatible_omegaconf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision={mixed_precision}. Expected one of ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_device(device: Optional[str]) -> str:
    if _is_none_like(device):
        device = str(get_default_device())
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to cpu.")
        return "cpu"
    return device


def _resolve_checkpoint(path_str: str) -> Path:
    ckpt = Path(path_str).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _search_upwards_for_file(start: Path, filename: str, *, max_levels: int = 6) -> Optional[Path]:
    current = start.resolve()
    for _ in range(max_levels + 1):
        candidate = current / filename
        if candidate.exists():
            return candidate.resolve()
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_config_path(ckpt_path: Path, explicit: Optional[str]) -> Path:
    if not _is_none_like(explicit):
        config_path = Path(str(explicit)).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path
    found = _search_upwards_for_file(ckpt_path.parent, "config.yaml")
    if found is None:
        raise FileNotFoundError(
            "Failed to locate `config.yaml` near checkpoint. "
            "Please pass `--config-path /path/to/config.yaml`."
        )
    return found


def _resolve_dataset_stats_path(ckpt_path: Path, explicit: Optional[str]) -> Path:
    if not _is_none_like(explicit):
        stats_path = Path(str(explicit)).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Dataset stats file not found: {stats_path}")
        return stats_path
    found = _search_upwards_for_file(ckpt_path.parent, "dataset_stats.json")
    if found is None:
        raise FileNotFoundError(
            "Failed to locate `dataset_stats.json` near checkpoint. "
            "Please pass `--dataset-stats-path /path/to/dataset_stats.json`."
        )
    return found


def _maybe_resolve_project_relative_path(value: Any) -> Any:
    if _is_none_like(value):
        return value
    text = str(value)
    if Path(text).is_absolute():
        return value
    for root in (Path.cwd(), Path(__file__).resolve().parent):
        candidate = (root / text).resolve()
        if candidate.exists():
            return str(candidate)
    return value


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _resize_rgb_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if pil_image.size != (width, height):
        pil_image = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(pil_image.convert("RGB"), dtype=np.uint8)


def _wrap_text(text: str, max_chars: int = 60) -> list[str]:
    words = text.strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _compose_real_robot_panel(
    *,
    task_description: str,
    current_frame: np.ndarray,
    pred_frames_by_offset: dict[int, np.ndarray],
    gt_frames_by_offset: dict[int, np.ndarray],
    panel_offsets: list[int],
) -> np.ndarray:
    target_h, target_w = current_frame.shape[:2]
    label_col_w = 190
    margin = 24
    gap = 12
    title_lines = _wrap_text(f'Task instruction: "{task_description}"', max_chars=72)
    title_h = 26 * len(title_lines) + 16
    row_header_h = 34
    row_h = target_h
    num_cols = len(panel_offsets)
    canvas_w = margin * 2 + label_col_w + num_cols * target_w + (num_cols - 1) * gap
    canvas_h = margin * 2 + title_h + row_header_h + 3 * row_h + 2 * gap + 24
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    text_x = margin
    text_y = margin
    for line in title_lines:
        draw.text((text_x, text_y), line, fill=(0, 0, 0))
        text_y += 26

    row_labels = [("Current Obs.", None), ("GT Future", None), ("Pred. Future", None)]
    header_y = margin + title_h
    base_y = header_y + row_header_h
    for row_idx, (row_label, _) in enumerate(row_labels):
        row_y = base_y + row_idx * (row_h + gap)
        draw.text((margin, row_y + 8), row_label, fill=(0, 0, 0))

    canvas.paste(Image.fromarray(current_frame), (margin + label_col_w, base_y))
    draw.text((margin + label_col_w + 8, header_y), "t=0", fill=(0, 0, 0))

    blank_tile = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    for col_idx, offset in enumerate(panel_offsets):
        x = margin + label_col_w + col_idx * (target_w + gap)
        draw.text((x + 8, header_y), f"+{offset}", fill=(0, 0, 0))
        gt = gt_frames_by_offset.get(offset)
        pred = pred_frames_by_offset.get(offset)
        gt_img = Image.fromarray(blank_tile if gt is None else gt)
        pred_img = Image.fromarray(blank_tile if pred is None else pred)
        canvas.paste(gt_img, (x, base_y + (row_h + gap)))
        canvas.paste(pred_img, (x, base_y + 2 * (row_h + gap)))
        if gt is None:
            draw.text((x + 12, base_y + (row_h + gap) + 12), "N/A", fill=(120, 120, 120))
        if pred is None:
            draw.text((x + 12, base_y + 2 * (row_h + gap) + 12), "N/A", fill=(120, 120, 120))

    return np.asarray(canvas, dtype=np.uint8)


def _parse_offsets(text: str) -> list[int]:
    offsets = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not offsets:
        raise ValueError("At least one panel offset is required.")
    return offsets


def _extract_request_index(path: Path) -> int:
    parts = path.name.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Unexpected request directory name: {path.name}")
    return int(parts[1])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(request_dir: Path, meta: dict[str, Any]) -> np.ndarray:
    state_json = request_dir / "state.json"
    if state_json.exists():
        return np.asarray(json.loads(state_json.read_text(encoding="utf-8")), dtype=np.float32)
    if "state" in meta:
        return np.asarray(meta["state"], dtype=np.float32)
    raise FileNotFoundError(f"Failed to locate state for request: {request_dir}")


def _load_current_frame(request_dir: Path) -> np.ndarray:
    image = Image.open(request_dir / "model_input_preview.png").convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _build_model_input_tensor(
    image: np.ndarray,
    *,
    video_size: tuple[int, int],
    resize_transform: ResizeSmallestSideAspectPreserving,
    crop_transform: CenterCrop,
    normalize_transform: Normalize,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    video = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
    if tuple(video.shape[-2:]) != tuple(video_size):
        video = resize_transform(video)
        video = crop_transform(video)
    video = normalize_transform(video)
    return video.to(device=device, dtype=dtype)


@torch.no_grad()
def _infer_future_video_with_latents(
    model: torch.nn.Module,
    *,
    prompt: Optional[str],
    input_image: torch.Tensor,
    num_video_frames: int,
    action_horizon: int,
    proprio: Optional[torch.Tensor],
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    num_inference_steps: int,
    sigma_shift: Optional[float],
    seed: Optional[int],
    rand_device: str,
    tiled: bool,
) -> dict[str, Any]:
    if not bool(getattr(model, "uses_state_fusion_action_expert")()):
        raise ValueError("Qualitative future-video export currently supports state-fusion Light-WAM only.")

    model.eval()
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
        raise ValueError(
            f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
        )

    _, _, height, width = input_image.shape
    checked_h, checked_w, checked_t = model._check_resize_height_width(height, width, num_video_frames)
    if (checked_h, checked_w) != (height, width):
        raise ValueError(
            f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
        )
    if checked_t != num_video_frames:
        raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)

    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    first_frame_latents = model._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    video_first_frame_latents = first_frame_latents
    apply_spatial_downsample = True
    restore_spatial_resolution = True
    if bool(getattr(model, "_use_lowres_video_training_objective")()):
        video_first_frame_latents, _ = model._maybe_downsample_video_latents_for_backbone(first_frame_latents)
        apply_spatial_downsample = False
        restore_spatial_resolution = False

    latent_t = (num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    latent_h = int(video_first_frame_latents.shape[-2])
    latent_w = int(video_first_frame_latents.shape[-1])

    generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    latents_video = torch.randn(
        (1, model.vae.model.z_dim, latent_t, latent_h, latent_w),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents_video[:, :, 0:1] = video_first_frame_latents.clone()

    use_prompt = prompt is not None
    use_context = context is not None or context_mask is not None
    if use_prompt and use_context:
        raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
    if not use_prompt and not use_context:
        raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

    if use_prompt:
        context, context_mask = model.encode_prompt(prompt)
    else:
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)

    if proprio is not None:
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    pred_action = model._predict_state_fusion_action_from_observation(
        observation_latents=first_frame_latents,
        action_horizon=action_horizon,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=fuse_flag,
    )

    infer_timesteps_video, infer_deltas_video = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents_video.dtype,
        shift_override=sigma_shift,
    )
    for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
        timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
        pred_video = model._predict_video_only(
            latents_video=latents_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=apply_spatial_downsample,
            restore_spatial_resolution=restore_spatial_resolution,
        )
        latents_video = model.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
        latents_video[:, :, 0:1] = video_first_frame_latents.clone()

    decoded_frames = model._decode_latents(latents_video, tiled=tiled)
    return {
        "action": pred_action[0].detach().to(device="cpu", dtype=torch.float32),
        "video": decoded_frames,
        "video_latents": latents_video[0].detach().to(device="cpu", dtype=torch.float32),
    }


def _build_prompt_context_cache(
    model: torch.nn.Module,
    task_description: str,
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = cache.get(task_description)
    if cached is not None:
        return cached
    prompt = DEFAULT_PROMPT.format(task=task_description)
    context, context_mask = model.encode_prompt(prompt)
    cached = (
        context.detach().to(device="cpu", dtype=torch.float32),
        context_mask.detach().to(device="cpu", dtype=torch.bool),
    )
    cache[task_description] = cached
    return cached


def _collect_rollout_request_dirs(rollout_dir: Path) -> list[Path]:
    request_dirs = [path for path in rollout_dir.iterdir() if path.is_dir() and path.name.startswith("request_")]
    if not request_dirs:
        raise ValueError(f"No request directories found under {rollout_dir}")
    return sorted(request_dirs, key=_extract_request_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Light-WAM real-robot rollout future-video qualitative panels.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint weights.")
    parser.add_argument("--rollout-dirs", nargs="+", required=True, help="One or more rollout request-log directories.")
    parser.add_argument("--output-dir", required=True, help="Directory to save exported qualitative data.")
    parser.add_argument("--config-path", default=None, help="Optional training config path. Auto-resolved near checkpoint by default.")
    parser.add_argument("--dataset-stats-path", default=None, help="Optional dataset stats path. Auto-resolved near checkpoint by default.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cuda or cpu.")
    parser.add_argument("--mixed-precision", default=None, help="Override mixed precision: no / fp16 / bf16.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Override future-video inference steps.")
    parser.add_argument("--sigma-shift", type=float, default=None, help="Optional sigma shift override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for future-video inference.")
    parser.add_argument("--rand-device", default="cpu", help="Torch generator device for future-video sampling.")
    parser.add_argument("--panel-offsets", default="8,16,24,32", help="Comma-separated future offsets shown in the panel.")
    parser.add_argument("--gt-offset", type=int, default=24, help="Future offset represented by the next rollout query.")
    parser.add_argument("--save-latents", action="store_true", help="Save predicted future latents.")
    parser.add_argument("--allow-missing-gt", action="store_true", help="Keep the final request even if +gt-offset future frame is unavailable.")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE encode/decode.")
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint(args.ckpt)
    config_path = _resolve_config_path(ckpt_path, args.config_path)
    dataset_stats_path = _resolve_dataset_stats_path(ckpt_path, args.dataset_stats_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_compatible_omegaconf(str(config_path))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.load_text_encoder = True
    if "action_dit_pretrained_path" in cfg.model:
        cfg.model.action_dit_pretrained_path = _maybe_resolve_project_relative_path(cfg.model.action_dit_pretrained_path)
    if "video_backbone_name" in cfg.model:
        cfg.model.video_backbone_name = _maybe_resolve_project_relative_path(cfg.model.video_backbone_name)

    effective_mixed_precision = str(cfg.get("mixed_precision", "bf16")) if _is_none_like(args.mixed_precision) else str(args.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(effective_mixed_precision)
    model_device = _resolve_device(args.device)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    model.load_checkpoint(str(ckpt_path))
    model = model.to(model_device).eval()

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: LightWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    video_size = (int(cfg.data.train.video_size[0]), int(cfg.data.train.video_size[1]))
    resize_transform = ResizeSmallestSideAspectPreserving(args={"img_w": video_size[1], "img_h": video_size[0]})
    crop_transform = CenterCrop(args={"img_w": video_size[1], "img_h": video_size[0]})
    normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

    panel_offsets = _parse_offsets(args.panel_offsets)
    request_offsets = [
        frame_idx * int(cfg.data.train.action_video_freq_ratio)
        for frame_idx in range((int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1)
    ]
    context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    summary_rollouts: list[dict[str, Any]] = []

    logger.info("Loaded Light-WAM qualitative real-robot exporter")
    logger.info("  ckpt=%s", ckpt_path)
    logger.info("  config=%s", config_path)
    logger.info("  dataset_stats=%s", dataset_stats_path)
    logger.info("  output_dir=%s", output_dir)
    logger.info("  rollout_dirs=%s", [str(Path(item).expanduser().resolve()) for item in args.rollout_dirs])

    for rollout_idx, rollout_path_str in enumerate(args.rollout_dirs):
        rollout_dir = Path(rollout_path_str).expanduser().resolve()
        request_dirs = _collect_rollout_request_dirs(rollout_dir)
        rollout_output_dir = output_dir / f"rollout_{rollout_idx + 1:02d}_{rollout_dir.name}"
        rollout_output_dir.mkdir(parents=True, exist_ok=True)
        panel_output_dir = rollout_output_dir / "qualitative_panels"
        panel_output_dir.mkdir(parents=True, exist_ok=True)

        clip_summaries: list[dict[str, Any]] = []
        for request_idx, request_dir in enumerate(request_dirs):
            next_request_dir = request_dirs[request_idx + 1] if request_idx + 1 < len(request_dirs) else None
            if next_request_dir is None and not args.allow_missing_gt:
                continue

            meta = _load_json(request_dir / "request_meta.json")
            task_description = str(meta["task_description"])
            request_info = meta.get("request_info", {})
            state = _load_state(request_dir, meta)
            current_frame = _load_current_frame(request_dir)
            gt_frames_by_offset: dict[int, np.ndarray] = {}
            if next_request_dir is not None:
                gt_frames_by_offset[int(args.gt_offset)] = _load_current_frame(next_request_dir)

            input_image = _build_model_input_tensor(
                current_frame,
                video_size=video_size,
                resize_transform=resize_transform,
                crop_transform=crop_transform,
                normalize_transform=normalize_transform,
                device=model.device,
                dtype=model.torch_dtype,
            )
            proprio = processor.normalizer.forward(
                processor.action_state_transform(
                    {"state": {processor.shape_meta["state"][0]["key"]: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
                )
            )["state"][processor.shape_meta["state"][0]["key"]].to(device=model.device, dtype=model.torch_dtype)

            context_cpu, context_mask_cpu = _build_prompt_context_cache(model, task_description, context_cache)
            future_payload = _infer_future_video_with_latents(
                model,
                prompt=None,
                input_image=input_image,
                num_video_frames=int(request_info.get("num_video_frames", len(request_offsets))),
                action_horizon=int(request_info.get("action_horizon", int(cfg.data.train.num_frames) - 1)),
                proprio=proprio,
                context=context_cpu.to(device=model.device, dtype=model.torch_dtype),
                context_mask=context_mask_cpu.to(device=model.device, dtype=torch.bool),
                num_inference_steps=int(
                    args.num_inference_steps
                    if args.num_inference_steps is not None
                    else request_info.get("num_inference_steps", int(cfg.get("eval_num_inference_steps", 20)))
                ),
                sigma_shift=args.sigma_shift if args.sigma_shift is not None else request_info.get("sigma_shift"),
                seed=args.seed,
                rand_device=str(args.rand_device),
                tiled=bool(args.tiled),
            )

            pred_frames_all = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in future_payload["video"]]
            pred_frames_by_offset: dict[int, np.ndarray] = {}
            for frame_idx, offset in enumerate(request_offsets):
                if offset not in panel_offsets:
                    continue
                pred_frames_by_offset[int(offset)] = _resize_rgb_array(
                    pred_frames_all[frame_idx],
                    current_frame.shape[1],
                    current_frame.shape[0],
                )

            clip_dir = rollout_output_dir / request_dir.name
            clip_dir.mkdir(parents=True, exist_ok=True)
            current_resized = _resize_rgb_array(current_frame, current_frame.shape[1], current_frame.shape[0])
            _save_rgb(clip_dir / "current_frame.png", current_resized)
            for offset, pred_frame in pred_frames_by_offset.items():
                _save_rgb(clip_dir / f"pred_frame_step{offset:03d}.png", pred_frame)
            for offset, gt_frame in gt_frames_by_offset.items():
                gt_resized = _resize_rgb_array(gt_frame, current_frame.shape[1], current_frame.shape[0])
                gt_frames_by_offset[offset] = gt_resized
                _save_rgb(clip_dir / f"gt_frame_step{offset:03d}.png", gt_resized)
                pred_for_offset = pred_frames_by_offset.get(offset)
                if pred_for_offset is not None:
                    triptych = np.concatenate([current_resized, pred_for_offset, gt_resized], axis=1)
                    _save_rgb(clip_dir / f"triplet_step{offset:03d}.png", triptych)

            if args.save_latents:
                torch.save(future_payload["video_latents"], clip_dir / "pred_video_latents.pt")

            panel = _compose_real_robot_panel(
                task_description=task_description,
                current_frame=current_resized,
                pred_frames_by_offset=pred_frames_by_offset,
                gt_frames_by_offset=gt_frames_by_offset,
                panel_offsets=panel_offsets,
            )
            _save_rgb(clip_dir / "qualitative_panel.png", panel)
            _save_rgb(panel_output_dir / f"{request_dir.name}.png", panel)

            clip_meta = {
                "request_dir": str(request_dir),
                "task_description": task_description,
                "panel_offsets": panel_offsets,
                "available_gt_offsets": sorted(int(item) for item in gt_frames_by_offset.keys()),
                "request_info": request_info,
                "predicted_offsets": sorted(int(item) for item in pred_frames_by_offset.keys()),
                "next_request_dir": None if next_request_dir is None else str(next_request_dir),
            }
            (clip_dir / "meta.json").write_text(json.dumps(clip_meta, indent=2), encoding="utf-8")
            clip_summaries.append(clip_meta)

        rollout_summary = {
            "rollout_dir": str(rollout_dir),
            "num_requests": len(request_dirs),
            "num_exported_clips": len(clip_summaries),
            "clips": clip_summaries,
        }
        (rollout_output_dir / "summary.json").write_text(json.dumps(rollout_summary, indent=2), encoding="utf-8")
        summary_rollouts.append(rollout_summary)

    summary = {
        "checkpoint": str(ckpt_path),
        "config_path": str(config_path),
        "dataset_stats_path": str(dataset_stats_path),
        "output_dir": str(output_dir),
        "panel_offsets": panel_offsets,
        "gt_offset": int(args.gt_offset),
        "rollouts": summary_rollouts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved real-robot qualitative summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
