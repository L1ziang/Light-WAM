import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from lightwam.datasets.lerobot.processors.lightwam_processor import LightWAMProcessor
from lightwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from lightwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from lightwam.utils.config_compat import load_compatible_omegaconf
from lightwam.utils.pytorch_utils import device_backend, get_default_device, set_global_seed, synchronize_device
from lightwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _sync_eval_cuda(device: str) -> None:
    synchronize_device(device)


def _maybe_preencode_task_prompt_context(
    task_description: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    *,
    model_device: str,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not bool(cfg.EVALUATION.get("preencode_prompt_context", False)):
        return None, None

    prompt = DEFAULT_PROMPT.format(task=task_description)
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        raise ValueError(
            "`EVALUATION.preencode_prompt_context=true` requires a loaded text encoder for the one-time prompt encoding step."
        )
    text_encoder.to(model_device)
    with torch.no_grad():
        context, context_mask = model.encode_prompt(prompt)
    cached_context = context.detach().to(device="cpu")
    cached_context_mask = context_mask.detach().to(device="cpu")

    text_encoder.to("cpu")
    _sync_eval_cuda(model_device)
    backend = device_backend(model_device)
    if backend is not None:
        backend.empty_cache()
    logging.info(
        "Pre-encoded task prompt context once and offloaded text encoder to CPU for rollout: task=%s",
        task_description,
    )
    return cached_context, cached_context_mask


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return str(get_default_device())


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _resolve_training_config_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("training_config_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "config.yaml")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate training config.yaml. Tried explicit "
        "EVALUATION.training_config_path and checkpoint parent directories. "
        "Please pass EVALUATION.training_config_path=/path/to/config.yaml."
    )
    raise FileNotFoundError(msg)


def _collect_relevant_cli_overrides() -> list[str]:
    try:
        hydra_overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        return []
    overrides = []
    for raw_override in hydra_overrides:
        key = raw_override.split("=", 1)[0].lstrip("+~")
        if key.startswith("model.") or key.startswith("data.") or key in {
            "mixed_precision",
            "eval_num_inference_steps",
        }:
            overrides.append(raw_override.lstrip("+"))
    return overrides


def _maybe_apply_training_run_config(cfg: DictConfig) -> DictConfig:
    if not bool(cfg.EVALUATION.get("use_training_run_config", False)):
        return cfg

    training_config_path = _resolve_training_config_path(cfg)
    train_cfg = load_compatible_omegaconf(str(training_config_path))
    merged_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

    if train_cfg.get("model") is not None:
        merged_cfg.model = OmegaConf.create(OmegaConf.to_container(train_cfg.model, resolve=False))
    if train_cfg.get("data") is not None:
        merged_cfg.data = OmegaConf.create(OmegaConf.to_container(train_cfg.data, resolve=False))
    if train_cfg.get("mixed_precision") is not None:
        merged_cfg.mixed_precision = train_cfg.mixed_precision
    if train_cfg.get("eval_num_inference_steps") is not None:
        merged_cfg.eval_num_inference_steps = train_cfg.eval_num_inference_steps

    # Training commonly disables text encoder because it uses cached embeddings.
    # LIBERO online evaluation needs prompt encoding, so force it back on.
    if merged_cfg.model.get("load_text_encoder") is not None:
        merged_cfg.model.load_text_encoder = True

    for override in _collect_relevant_cli_overrides():
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.from_cli([override]))

    if merged_cfg.model.get("load_text_encoder") is not None:
        merged_cfg.model.load_text_encoder = True

    logging.info("Loaded training run config for evaluation: %s", training_config_path)
    logging.info(
        "Effective eval model: target=%s use_wam_adapter=%s remove_original_action_expert=%s",
        merged_cfg.model.get("_target_"),
        merged_cfg.model.get("wam_adapter", {}).get("use_wam_adapter", None),
        merged_cfg.model.get("wam_adapter", {}).get("remove_original_action_expert", None),
    )
    return merged_cfg


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _build_eval_image_tensor_like_training(
    imgs: dict[str, np.ndarray],
    cfg: DictConfig,
    processor: LightWAMProcessor,
    input_w: int,
    input_h: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply the same image transform chain used by the training dataset."""
    image_batch = {}
    for meta in processor.shape_meta["images"][: int(processor.num_output_cameras)]:
        key = meta["key"]
        if key not in imgs:
            raise KeyError(f"LIBERO observation image `{key}` is missing. Available keys: {sorted(imgs.keys())}")
        image_batch[key] = torch.as_tensor(imgs[key], dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

    # Reuse LightWAMProcessor val_transforms exactly: ToTensor + per-camera Resize.
    pixel_values = processor.build_pixel_values_from_episode_images({"images": image_batch})
    if pixel_values.ndim != 5:
        raise ValueError(f"Expected pixel_values [N,T,C,H,W], got {tuple(pixel_values.shape)}")

    num_cameras = int(pixel_values.shape[0])
    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    if num_cameras == 1:
        video = pixel_values[0]
    elif concatenation == "horizontal":
        video = torch.cat([pixel_values[i] for i in range(num_cameras)], dim=-1)
    elif concatenation == "vertical":
        video = torch.cat([pixel_values[i] for i in range(num_cameras)], dim=-2)
    else:
        raise ValueError(f"Invalid concat_multi_camera: {concatenation}")

    resize_transform = ResizeSmallestSideAspectPreserving(args={"img_w": input_w, "img_h": input_h})
    crop_transform = CenterCrop(args={"img_w": input_w, "img_h": input_h})
    normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
    video = normalize_transform(crop_transform(resize_transform(video)))

    if tuple(video.shape) != (1, 3, input_h, input_w):
        raise ValueError(
            "Training-aligned eval image shape mismatch: "
            f"got {tuple(video.shape)}, expected {(1, 3, input_h, input_w)}."
        )
    return video.to(device=device, dtype=dtype)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: LightWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: LightWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    num_cameras = int(processor.num_output_cameras)
    if num_cameras not in {1, 2}:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    x = _build_eval_image_tensor_like_training(
        imgs=imgs,
        cfg=cfg,
        processor=processor,
        input_w=width,
        input_h=height,
        device=device,
        dtype=dtype,
    )

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: LightWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: LightWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    latency_log_context: Optional[str] = None,
    cached_prompt_context: Optional[torch.Tensor] = None,
    cached_prompt_context_mask: Optional[torch.Tensor] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt = None
    if cached_prompt_context is None or cached_prompt_context_mask is None:
        prompt_template = DEFAULT_PROMPT
        prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if cached_prompt_context is not None or cached_prompt_context_mask is not None:
        if cached_prompt_context is None or cached_prompt_context_mask is None:
            raise ValueError(
                "`cached_prompt_context` and `cached_prompt_context_mask` must be provided together."
            )
        infer_kwargs["context"] = cached_prompt_context
        infer_kwargs["context_mask"] = cached_prompt_context_mask
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    print_action_latency = bool(cfg.EVALUATION.get("print_action_latency", False))
    print_action_latency_sync_cuda = bool(cfg.EVALUATION.get("print_action_latency_sync_cuda", True))
    predicted_future_frames = None
    infer_latency_ms = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if print_action_latency and print_action_latency_sync_cuda:
            _sync_eval_cuda(model_device)
        infer_start = time.perf_counter()
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
        if print_action_latency and print_action_latency_sync_cuda:
            _sync_eval_cuda(model_device)
        infer_latency_ms = (time.perf_counter() - infer_start) * 1000.0
    if print_action_latency and infer_latency_ms is not None:
        context_prefix = "" if latency_log_context is None else f"{latency_log_context} "
        print(
            f"[action-latency] {context_prefix}overall_latency_ms={infer_latency_ms:.3f}",
            flush=True,
        )
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _safe_close_env(env) -> None:
    close_fn = getattr(env, "close", None)
    if not callable(close_fn):
        return
    try:
        close_fn()
    except Exception as exc:
        logging.warning("Ignoring environment close error during cleanup: %s", exc)


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: LightWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    cached_prompt_context: Optional[torch.Tensor] = None,
    cached_prompt_context_mask: Optional[torch.Tensor] = None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    query_idx = 0

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            query_idx += 1
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                latency_log_context=f"episode={episode_idx + 1} step={t} query={query_idx}",
                cached_prompt_context=cached_prompt_context,
                cached_prompt_context_mask=cached_prompt_context_mask,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: LightWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    cached_prompt_context, cached_prompt_context_mask = _maybe_preencode_task_prompt_context(
        task_description,
        model,
        cfg,
        model_device=model_device,
    )
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    try:
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                cached_prompt_context=cached_prompt_context,
                cached_prompt_context_mask=cached_prompt_context_mask,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
            if visualize_future_video:
                results["episode_future_video_psnr"].append(episode_mean_psnr)

            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
            if visualize_future_video:
                if len(predicted_future_video_clips) == 0:
                    logging.warning(
                        "No predicted future frames collected for task %s trial %s.",
                        cfg.EVALUATION.task_id,
                        trial_idx,
                    )
                else:
                    all_gt_frames = []
                    all_pred_frames = []
                    for clip in predicted_future_video_clips:
                        all_gt_frames.extend(clip["gt_frames"])
                        all_pred_frames.extend(clip["pred_frames"])
                        save_prediction_video(
                            predicted_video_dir,
                            clip["gt_frames"],
                            clip["pred_frames"],
                            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                            clip["replan_idx"],
                            success=success,
                            task_description=task_description,
                        )
                    save_prediction_video(
                        predicted_video_dir,
                        all_gt_frames,
                        all_pred_frames,
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        "all",
                        success=success,
                        task_description=task_description,
                    )
    finally:
        _safe_close_env(env)

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    cfg = _maybe_apply_training_run_config(cfg)
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use scripts/eval_libero.sh or an external scheduler for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    logging.info("Evaluating on device: %s (first parameter on %s)", model_device, next(model.parameters()).device)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: LightWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
