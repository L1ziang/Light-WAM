import argparse
import asyncio
import base64
import functools
import http
import inspect
import json
import logging
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass

import msgpack
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames as websocket_frames
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lightwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from lightwam.datasets.lerobot.processors.lightwam_processor import LightWAMProcessor
from lightwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from lightwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from lightwam.utils.config_compat import load_compatible_omegaconf
from lightwam.utils.pytorch_utils import get_default_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GripperChannelCalibration:
    index: int
    min_value: float
    max_value: float
    close_snap_threshold: float


def msgpack_pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def msgpack_unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


MsgpackPacker = functools.partial(msgpack.Packer, default=msgpack_pack_array)
msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=msgpack_unpack_array)


class WebsocketPolicyServer:
    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("Websocket client connected: %s", websocket.remote_address)
        packer = MsgpackPacker()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_unpackb(await websocket.recv())

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_elapsed = time.monotonic() - infer_start

                action["server_timing"] = {"infer_ms": infer_elapsed * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Websocket client disconnected: %s", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websocket_frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_json_dumps(data: Any) -> bytes:
    return json.dumps(data, cls=NumpyJSONEncoder).encode("utf-8")


def numpy_json_loads(payload: bytes) -> Any:
    def object_hook(obj: dict[str, Any]) -> Any:
        if obj.get("__numpy_array__") is True:
            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        return obj

    return json.loads(payload.decode("utf-8"), object_hook=object_hook)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        chunk = sock.recv(min(size - received, 4096))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value}")


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


class RealRobotPolicyService:
    def __init__(
        self,
        *,
        ckpt_path: Path,
        config_path: Path,
        dataset_stats_path: Path,
        device: str,
        mixed_precision: Optional[str],
        action_horizon: Optional[int],
        num_inference_steps: Optional[int],
        sigma_shift: Optional[float],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        binarize_gripper: bool,
        use_prompt_cache: bool,
        query_save_dir: Optional[str],
        save_queries: bool,
        gripper_dim_indices: tuple[int, ...],
        enable_real_gripper_snap: bool,
        real_gripper_close_snap_threshold: float,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.dataset_stats_path = dataset_stats_path
        self.device = _resolve_device(device)

        self.cfg = load_compatible_omegaconf(str(self.config_path))
        self.cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        if not hasattr(self.cfg, "model") or not hasattr(self.cfg, "data"):
            raise ValueError(f"Invalid config payload in {self.config_path}. Missing `model` or `data`.")

        self.cfg.model.load_text_encoder = True
        if "action_dit_pretrained_path" in self.cfg.model:
            self.cfg.model.action_dit_pretrained_path = _maybe_resolve_project_relative_path(
                self.cfg.model.action_dit_pretrained_path
            )
        if "video_backbone_name" in self.cfg.model:
            self.cfg.model.video_backbone_name = _maybe_resolve_project_relative_path(
                self.cfg.model.video_backbone_name
            )

        saved_mixed_precision = str(self.cfg.get("mixed_precision", "bf16"))
        effective_mixed_precision = (
            saved_mixed_precision if _is_none_like(mixed_precision) else str(mixed_precision)
        )
        self.model_dtype = _mixed_precision_to_model_dtype(effective_mixed_precision)

        self.model = instantiate(self.cfg.model, model_dtype=self.model_dtype, device=self.device)
        self.model.load_checkpoint(str(self.ckpt_path))
        self.model = self.model.to(self.device).eval()

        dataset_stats = load_dataset_stats_from_json(str(self.dataset_stats_path))
        self.processor: LightWAMProcessor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(dataset_stats)

        video_size = self.cfg.data.train.get("video_size", [384, 320])
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.num_video_frames = (int(self.cfg.data.train.num_frames) - 1) // int(
            self.cfg.data.train.action_video_freq_ratio
        ) + 1
        self.action_horizon = (
            int(self.cfg.data.train.num_frames) - 1 if action_horizon is None else int(action_horizon)
        )
        self.num_inference_steps = (
            int(self.cfg.get("eval_num_inference_steps", 20))
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        self.sigma_shift = sigma_shift
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.binarize_gripper = bool(binarize_gripper)
        self.use_prompt_cache = bool(use_prompt_cache)
        self.concat_multi_camera = str(self.cfg.data.train.get("concat_multi_camera", "robotwin"))
        self.save_queries = bool(save_queries)
        self.enable_real_gripper_snap = bool(enable_real_gripper_snap)
        self.real_gripper_close_snap_threshold = float(real_gripper_close_snap_threshold)
        self.query_save_dir = (
            Path(str(query_save_dir)).expanduser().resolve()
            if (query_save_dir is not None and not _is_none_like(query_save_dir))
            else None
        )
        if self.save_queries and self.query_save_dir is not None:
            self.query_save_dir.mkdir(parents=True, exist_ok=True)

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self._infer_lock = threading.Lock()
        self._request_count = 0
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._gripper_calibrations = self._build_gripper_calibrations(gripper_dim_indices)

        logger.info("Loaded real-robot Light-WAM policy service")
        logger.info("  ckpt=%s", self.ckpt_path)
        logger.info("  config=%s", self.config_path)
        logger.info("  dataset_stats=%s", self.dataset_stats_path)
        logger.info("  device=%s dtype=%s", self.device, self.model_dtype)
        logger.info("  action_horizon=%d num_video_frames=%d", self.action_horizon, self.num_video_frames)
        logger.info("  save_queries=%s query_save_dir=%s", self.save_queries, self.query_save_dir)
        logger.info(
            "  enable_real_gripper_snap=%s gripper_dims=%s close_snap_threshold=%.3f",
            self.enable_real_gripper_snap,
            [item.index for item in self._gripper_calibrations],
            self.real_gripper_close_snap_threshold,
        )

    def _build_gripper_calibrations(self, gripper_dim_indices: tuple[int, ...]) -> list[GripperChannelCalibration]:
        if not gripper_dim_indices:
            return []
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        stats = self.processor.normalizer.normalizers["action"][action_key].get_stats()
        action_min = stats["min"].detach().to(dtype=torch.float32, device="cpu").numpy()
        action_max = stats["max"].detach().to(dtype=torch.float32, device="cpu").numpy()
        action_dim = int(action_min.shape[-1])

        calibrations: list[GripperChannelCalibration] = []
        for raw_index in gripper_dim_indices:
            index = int(raw_index)
            if index < 0 or index >= action_dim:
                logger.warning(
                    "Skip out-of-range gripper dim index=%d for action_dim=%d",
                    index,
                    action_dim,
                )
                continue
            min_value = float(action_min[index])
            max_value = float(action_max[index])
            if max_value < min_value:
                min_value, max_value = max_value, min_value
            calibrations.append(
                GripperChannelCalibration(
                    index=index,
                    min_value=min_value,
                    max_value=max_value,
                    close_snap_threshold=self.real_gripper_close_snap_threshold,
                )
            )
        return calibrations

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key].to(device=self.model.device, dtype=self.model.torch_dtype)

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    @staticmethod
    def _finalize_action_chunk(
        action_chunk: np.ndarray,
        *,
        apply_gripper_postprocess: bool,
        binarize_gripper: bool,
    ) -> np.ndarray:
        output = np.asarray(action_chunk, dtype=np.float32).copy()
        if not apply_gripper_postprocess:
            return output
        output[..., -1] = output[..., -1] * 2 - 1
        if binarize_gripper:
            output[..., -1] = np.sign(output[..., -1])
        return output

    def _apply_real_gripper_snap(self, action_chunk: np.ndarray) -> np.ndarray:
        output = np.asarray(action_chunk, dtype=np.float32).copy()
        if not self.enable_real_gripper_snap or not self._gripper_calibrations:
            return output
        for calibration in self._gripper_calibrations:
            channel = np.clip(output[..., calibration.index], calibration.min_value, calibration.max_value)
            channel = np.round(channel)
            channel = np.where(
                channel <= calibration.min_value + calibration.close_snap_threshold,
                calibration.min_value,
                channel,
            )
            output[..., calibration.index] = channel.astype(np.float32)
        return output

    @staticmethod
    def _require_image_array(images: dict[str, Any], key: str) -> np.ndarray:
        if key not in images:
            raise KeyError(f"Missing image key `{key}` in observation.images.")
        array = np.asarray(images[key])
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Image `{key}` must be HWC RGB, got shape {tuple(array.shape)}")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array.copy())

    def _build_input_image_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
        images = observation.get("images")
        if not isinstance(images, dict):
            raise ValueError("`observation.images` must be a dict of camera arrays.")

        episode_images = {
            "images": {
                "cam_high": torch.from_numpy(self._require_image_array(images, "cam_high")).permute(2, 0, 1).unsqueeze(0),
                "cam_left_wrist": torch.from_numpy(self._require_image_array(images, "cam_left_wrist")).permute(2, 0, 1).unsqueeze(0),
                "cam_right_wrist": torch.from_numpy(self._require_image_array(images, "cam_right_wrist")).permute(2, 0, 1).unsqueeze(0),
            }
        }
        pixel_values = self.processor.build_pixel_values_from_episode_images(episode_images)
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected pixel_values [N,T,C,H,W], got {tuple(pixel_values.shape)}")
        if self.concat_multi_camera != "robotwin":
            raise ValueError(
                f"Real-robot service expects concat_multi_camera='robotwin', got {self.concat_multi_camera}"
            )
        if int(self.processor.num_output_cameras) != 3:
            raise ValueError(f"Real-robot service requires num_output_cameras=3, got {self.processor.num_output_cameras}")

        cam_top = transforms_F.resize(
            pixel_values[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_left = transforms_F.resize(
            pixel_values[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_right = transforms_F.resize(
            pixel_values[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        bottom = torch.cat([cam_left, cam_right], dim=-1)
        video = torch.cat([cam_top, bottom], dim=-2)
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        if tuple(video.shape) != (1, 3, self.video_size[0], self.video_size[1]):
            raise ValueError(
                "Training-aligned real-robot image shape mismatch: "
                f"got {tuple(video.shape)}, expected {(1, 3, self.video_size[0], self.video_size[1])}."
            )
        return video.to(device=self.model.device, dtype=self.model.torch_dtype)

    @staticmethod
    def _save_rgb(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image_uint8 = np.asarray(image, dtype=np.uint8)
        Image.fromarray(image_uint8).save(path)

    @staticmethod
    def _tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
        tensor = video.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected [1,C,H,W], got {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W], got {tuple(tensor.shape)}")
        tensor = tensor.clamp(-1.0, 1.0)
        tensor = ((tensor * 0.5) + 0.5).clamp(0.0, 1.0)
        image = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return image

    @staticmethod
    def _build_robotwin_canvas_from_images(images: dict[str, np.ndarray]) -> np.ndarray:
        cam_top = np.asarray(
            Image.fromarray(images["cam_high"]).resize((320, 256), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
        cam_left = np.asarray(
            Image.fromarray(images["cam_left_wrist"]).resize((160, 128), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
        cam_right = np.asarray(
            Image.fromarray(images["cam_right_wrist"]).resize((160, 128), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
        canvas = np.full((384, 320, 3), 255, dtype=np.uint8)
        canvas[0:256, :, :] = cam_top
        canvas[256:384, 0:160, :] = cam_left
        canvas[256:384, 160:320, :] = cam_right
        return canvas

    def _maybe_dump_three_view_request(
        self,
        *,
        task_description: str,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        model_input_image: torch.Tensor,
        request_info: dict[str, Any],
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.save_queries or self.query_save_dir is None:
            return

        request_index = int(self._request_count)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_dir = self.query_save_dir / f"request_{request_index:06d}_{timestamp}"
        request_dir.mkdir(parents=True, exist_ok=True)

        for key, image in images.items():
            self._save_rgb(request_dir / f"{key}.png", image)

        canvas = self._build_robotwin_canvas_from_images(images)
        self._save_rgb(request_dir / "robotwin_canvas.png", canvas)
        self._save_rgb(request_dir / "model_input_preview.png", self._tensor_video_to_uint8(model_input_image))

        (request_dir / "task.txt").write_text(str(task_description), encoding="utf-8")
        (request_dir / "state.json").write_text(json.dumps(np.asarray(state, dtype=np.float32).tolist(), indent=2), encoding="utf-8")
        np.save(request_dir / "state.npy", np.asarray(state, dtype=np.float32))

        payload = {
            "task_description": str(task_description),
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "request_info": request_info,
        }
        if result is not None:
            payload["result"] = result
            if "action_chunk" in result:
                np.save(request_dir / "response_action_chunk.npy", np.asarray(result["action_chunk"], dtype=np.float32))
        (request_dir / "request_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _maybe_dump_canvas_request(
        self,
        *,
        task_description: str,
        state: np.ndarray,
        current_obs_image: np.ndarray,
        model_input_image: torch.Tensor,
        request_info: dict[str, Any],
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.save_queries or self.query_save_dir is None:
            return

        request_index = int(self._request_count)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_dir = self.query_save_dir / f"request_{request_index:06d}_{timestamp}"
        request_dir.mkdir(parents=True, exist_ok=True)

        self._save_rgb(request_dir / "current_obs_image.png", current_obs_image)
        self._save_rgb(request_dir / "model_input_preview.png", self._tensor_video_to_uint8(model_input_image))
        (request_dir / "task.txt").write_text(str(task_description), encoding="utf-8")
        (request_dir / "state.json").write_text(json.dumps(np.asarray(state, dtype=np.float32).tolist(), indent=2), encoding="utf-8")
        np.save(request_dir / "state.npy", np.asarray(state, dtype=np.float32))

        payload = {
            "task_description": str(task_description),
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "request_info": request_info,
        }
        if result is not None:
            payload["result"] = result
            if "action_chunk" in result:
                np.save(request_dir / "response_action_chunk.npy", np.asarray(result["action_chunk"], dtype=np.float32))
        (request_dir / "request_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _require_canvas_array(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"`current_obs_image` must be HWC RGB, got shape {tuple(array.shape)}")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array.copy())

    def _build_input_image_tensor_from_canvas(self, current_obs_image: Any) -> torch.Tensor:
        image = self._require_canvas_array(current_obs_image)
        video = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        if tuple(video.shape) != (1, 3, self.video_size[0], self.video_size[1]):
            raise ValueError(
                "Canvas-image real-robot shape mismatch after preprocessing: "
                f"got {tuple(video.shape)}, expected {(1, 3, self.video_size[0], self.video_size[1])}."
            )
        return video.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _get_cached_context(self, task_description: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._prompt_cache.get(task_description)
        if cached is not None:
            return cached
        prompt = DEFAULT_PROMPT.format(task=task_description)
        with self._infer_lock, torch.no_grad():
            context, context_mask = self.model.encode_prompt(prompt)
        cached = (
            context.detach().to(device="cpu", dtype=torch.float32),
            context_mask.detach().to(device="cpu", dtype=torch.bool),
        )
        self._prompt_cache[task_description] = cached
        return cached

    def health(self, _payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint_path": str(self.ckpt_path),
            "config_path": str(self.config_path),
            "dataset_stats_path": str(self.dataset_stats_path),
            "device": str(self.device),
            "torch_dtype": str(self.model_dtype),
            "action_horizon": int(self.action_horizon),
            "num_video_frames": int(self.num_video_frames),
            "num_inference_steps": int(self.num_inference_steps),
            "sigma_shift": self.sigma_shift,
            "binarize_gripper": bool(self.binarize_gripper),
            "enable_real_gripper_snap": bool(self.enable_real_gripper_snap),
            "real_gripper_dim_indices": [item.index for item in self._gripper_calibrations],
            "real_gripper_close_snap_threshold": float(self.real_gripper_close_snap_threshold),
            "use_prompt_cache": bool(self.use_prompt_cache),
            "cached_prompt_count": int(len(self._prompt_cache)),
            "request_count": int(self._request_count),
        }

    def infer_action_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("`payload` must be a JSON object.")
        task_description = payload.get("task_description")
        observation = payload.get("observation", payload.get("obs"))
        if not isinstance(task_description, str) or task_description.strip() == "":
            raise ValueError("`task_description` must be a non-empty string.")
        if not isinstance(observation, dict):
            raise ValueError("`observation` must be a JSON object with `images` and `state`.")

        state = observation.get("state")
        if state is None:
            raise ValueError("`observation.state` is required.")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"`observation.state` must have shape (14,), got {tuple(state.shape)}")

        action_horizon = self.action_horizon if payload.get("action_horizon") is None else int(payload["action_horizon"])
        num_inference_steps = (
            self.num_inference_steps
            if payload.get("num_inference_steps") is None
            else int(payload["num_inference_steps"])
        )
        sigma_shift = _parse_optional_float(payload.get("sigma_shift"))
        if sigma_shift is None:
            sigma_shift = self.sigma_shift
        seed = _parse_optional_int(payload.get("seed"))
        text_cfg_scale = float(payload.get("text_cfg_scale", self.text_cfg_scale))
        negative_prompt = str(payload.get("negative_prompt", self.negative_prompt))
        rand_device = str(payload.get("rand_device", self.rand_device))
        tiled = bool(payload.get("tiled", self.tiled))
        binarize_gripper = self.binarize_gripper if payload.get("binarize_gripper") is None else _parse_bool(
            payload.get("binarize_gripper")
        )
        apply_gripper_postprocess = True if payload.get("apply_gripper_postprocess") is None else _parse_bool(
            payload.get("apply_gripper_postprocess")
        )
        apply_real_gripper_snap = (
            self.enable_real_gripper_snap
            if payload.get("apply_real_gripper_snap") is None
            else _parse_bool(payload.get("apply_real_gripper_snap"))
        )
        apply_real_gripper_snap = (
            self.enable_real_gripper_snap
            if payload.get("apply_real_gripper_snap") is None
            else _parse_bool(payload.get("apply_real_gripper_snap"))
        )

        image_tensor = self._build_input_image_tensor(observation)
        proprio = self._normalize_state(state)

        infer_kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=task_description),
            "input_image": image_tensor,
            "action_horizon": int(action_horizon),
            "proprio": proprio,
            "negative_prompt": negative_prompt,
            "text_cfg_scale": text_cfg_scale,
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": sigma_shift,
            "seed": seed,
            "rand_device": rand_device,
            "tiled": tiled,
        }
        if self.use_prompt_cache:
            context, context_mask = self._get_cached_context(task_description)
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = context.to(device=self.model.device, dtype=self.model.torch_dtype)
            infer_kwargs["context_mask"] = context_mask.to(device=self.model.device, dtype=torch.bool)
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self.num_video_frames)

        started = time.perf_counter()
        with self._infer_lock, torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        elapsed = time.perf_counter() - started
        self._request_count += 1

        raw_action_chunk = self._denormalize_action(pred["action"])[0]
        action_chunk = self._finalize_action_chunk(
            raw_action_chunk,
            apply_gripper_postprocess=apply_gripper_postprocess,
            binarize_gripper=binarize_gripper,
        )
        if apply_real_gripper_snap:
            action_chunk = self._apply_real_gripper_snap(action_chunk)
        output = {
            "action_chunk": np.asarray(action_chunk, dtype=np.float32),
            "apply_gripper_postprocess": bool(apply_gripper_postprocess),
            "apply_real_gripper_snap": bool(apply_real_gripper_snap),
            "latency_s": float(elapsed),
            "action_horizon": int(action_horizon),
            "task_description": task_description,
        }
        request_info = {
            "request_type": "three_view",
            "action_horizon": int(action_horizon),
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": sigma_shift,
            "text_cfg_scale": float(text_cfg_scale),
            "negative_prompt": str(negative_prompt),
            "rand_device": str(rand_device),
            "tiled": bool(tiled),
            "binarize_gripper": bool(binarize_gripper),
            "apply_gripper_postprocess": bool(apply_gripper_postprocess),
            "apply_real_gripper_snap": bool(apply_real_gripper_snap),
            "num_video_frames": int(self.num_video_frames),
        }
        self._maybe_dump_three_view_request(
            task_description=task_description,
            state=state,
            images={
                "cam_high": self._require_image_array(observation["images"], "cam_high"),
                "cam_left_wrist": self._require_image_array(observation["images"], "cam_left_wrist"),
                "cam_right_wrist": self._require_image_array(observation["images"], "cam_right_wrist"),
            },
            model_input_image=image_tensor,
            request_info=request_info,
            result={
                "latency_s": float(elapsed),
                "action_chunk_shape": list(np.asarray(action_chunk).shape),
                "first_action": np.asarray(action_chunk, dtype=np.float32)[0].tolist(),
                "action_chunk": np.asarray(action_chunk, dtype=np.float32).tolist(),
            },
        )
        return output

    def infer_action_chunk_from_canvas(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("`payload` must be a JSON object.")
        task_description = payload.get("task_description")
        observation = payload.get("observation", payload.get("obs"))
        if not isinstance(task_description, str) or task_description.strip() == "":
            raise ValueError("`task_description` must be a non-empty string.")
        if not isinstance(observation, dict):
            raise ValueError("`observation` must be a JSON object with `state`.")

        state = observation.get("state")
        if state is None:
            raise ValueError("`observation.state` is required.")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"`observation.state` must have shape (14,), got {tuple(state.shape)}")

        current_obs_image = payload.get("current_obs_image", observation.get("current_obs_image"))
        if current_obs_image is None:
            raise ValueError("`current_obs_image` is required for `infer_action_chunk_from_canvas`.")

        action_horizon = self.action_horizon if payload.get("action_horizon") is None else int(payload["action_horizon"])
        num_inference_steps = (
            self.num_inference_steps
            if payload.get("num_inference_steps") is None
            else int(payload["num_inference_steps"])
        )
        sigma_shift = _parse_optional_float(payload.get("sigma_shift"))
        if sigma_shift is None:
            sigma_shift = self.sigma_shift
        seed = _parse_optional_int(payload.get("seed"))
        text_cfg_scale = float(payload.get("text_cfg_scale", self.text_cfg_scale))
        negative_prompt = str(payload.get("negative_prompt", self.negative_prompt))
        rand_device = str(payload.get("rand_device", self.rand_device))
        tiled = bool(payload.get("tiled", self.tiled))
        binarize_gripper = self.binarize_gripper if payload.get("binarize_gripper") is None else _parse_bool(
            payload.get("binarize_gripper")
        )
        apply_gripper_postprocess = True if payload.get("apply_gripper_postprocess") is None else _parse_bool(
            payload.get("apply_gripper_postprocess")
        )

        image_tensor = self._build_input_image_tensor_from_canvas(current_obs_image)
        proprio = self._normalize_state(state)

        infer_kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=task_description),
            "input_image": image_tensor,
            "action_horizon": int(action_horizon),
            "proprio": proprio,
            "negative_prompt": negative_prompt,
            "text_cfg_scale": text_cfg_scale,
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": sigma_shift,
            "seed": seed,
            "rand_device": rand_device,
            "tiled": tiled,
        }
        if self.use_prompt_cache:
            context, context_mask = self._get_cached_context(task_description)
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = context.to(device=self.model.device, dtype=self.model.torch_dtype)
            infer_kwargs["context_mask"] = context_mask.to(device=self.model.device, dtype=torch.bool)
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self.num_video_frames)

        started = time.perf_counter()
        with self._infer_lock, torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        elapsed = time.perf_counter() - started
        self._request_count += 1

        raw_action_chunk = self._denormalize_action(pred["action"])[0]
        action_chunk = self._finalize_action_chunk(
            raw_action_chunk,
            apply_gripper_postprocess=apply_gripper_postprocess,
            binarize_gripper=binarize_gripper,
        )
        if apply_real_gripper_snap:
            action_chunk = self._apply_real_gripper_snap(action_chunk)
        output = {
            "action_chunk": np.asarray(action_chunk, dtype=np.float32),
            "apply_gripper_postprocess": bool(apply_gripper_postprocess),
            "apply_real_gripper_snap": bool(apply_real_gripper_snap),
            "latency_s": float(elapsed),
            "action_horizon": int(action_horizon),
            "task_description": task_description,
        }
        request_info = {
            "request_type": "canvas",
            "action_horizon": int(action_horizon),
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": sigma_shift,
            "text_cfg_scale": float(text_cfg_scale),
            "negative_prompt": str(negative_prompt),
            "rand_device": str(rand_device),
            "tiled": bool(tiled),
            "binarize_gripper": bool(binarize_gripper),
            "apply_gripper_postprocess": bool(apply_gripper_postprocess),
            "apply_real_gripper_snap": bool(apply_real_gripper_snap),
            "num_video_frames": int(self.num_video_frames),
        }
        self._maybe_dump_canvas_request(
            task_description=task_description,
            state=state,
            current_obs_image=self._require_canvas_array(current_obs_image),
            model_input_image=image_tensor,
            request_info=request_info,
            result={
                "latency_s": float(elapsed),
                "action_chunk_shape": list(np.asarray(action_chunk).shape),
                "first_action": np.asarray(action_chunk, dtype=np.float32)[0].tolist(),
                "action_chunk": np.asarray(action_chunk, dtype=np.float32).tolist(),
            },
        )
        return output


class PolicySocketServer:
    def __init__(self, handler: RealRobotPolicyService, host: str, port: int) -> None:
        self.handler = handler
        self.host = str(host)
        self.port = int(port)
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.client_threads: list[threading.Thread] = []

    def start(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(16)
        self.running = True
        logger.info("Real-robot policy service listening on %s:%d", self.host, self.port)
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
            except OSError:
                if self.running:
                    raise
                break
            thread = threading.Thread(
                target=self._handle_client,
                args=(client_socket, address),
                daemon=True,
            )
            thread.start()
            self.client_threads.append(thread)

    def stop(self) -> None:
        self.running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        for thread in self.client_threads:
            thread.join(timeout=1.0)

    def _send_response(self, client_socket: socket.socket, payload: dict[str, Any]) -> None:
        encoded = numpy_json_dumps(payload)
        client_socket.sendall(len(encoded).to_bytes(4, "big"))
        client_socket.sendall(encoded)

    def _handle_client(self, client_socket: socket.socket, address: tuple[str, int]) -> None:
        logger.info("Client connected: %s:%s", address[0], address[1])
        with client_socket:
            while self.running:
                try:
                    header = client_socket.recv(4)
                    if not header:
                        return
                    payload_len = int.from_bytes(header, "big")
                    request = numpy_json_loads(_recv_exact(client_socket, payload_len))
                    command = request.get("cmd")
                    payload = request.get("payload")
                    if not isinstance(command, str) or command.strip() == "":
                        raise ValueError("Request must contain a non-empty string `cmd`.")
                    method = getattr(self.handler, command, None)
                    if not callable(method):
                        raise AttributeError(f"Unknown command: {command}")
                    result = method(payload)
                    self._send_response(client_socket, {"ok": True, "result": result})
                except Exception as exc:
                    logger.exception("Request handling failed")
                    self._send_response(
                        client_socket,
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    return


class LightWAMOpenPIAdapter:
    def __init__(self, service: RealRobotPolicyService) -> None:
        self.service = service

    @staticmethod
    def _to_hwc_rgb(image: Any, camera_name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Image `{camera_name}` must be 3D, got shape {tuple(array.shape)}")
        if array.shape[0] in (1, 3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
        elif array.shape[-1] >= 3:
            array = array[..., :3]
        else:
            raise ValueError(f"Image `{camera_name}` must be CHW or HWC RGB, got shape {tuple(array.shape)}")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            raise ValueError("OpenPI observation must be a dict.")
        task_description = obs.get("prompt", obs.get("task_description"))
        if not isinstance(task_description, str) or task_description.strip() == "":
            raise ValueError("OpenPI observation must contain non-empty `prompt` or `task_description`.")
        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError("OpenPI observation must contain `images` dict.")

        payload: dict[str, Any] = {
            "task_description": task_description.strip(),
            "observation": {
                "state": np.asarray(obs.get("state"), dtype=np.float32),
                "images": {
                    "cam_high": self._to_hwc_rgb(images["cam_high"], "cam_high"),
                    "cam_left_wrist": self._to_hwc_rgb(images["cam_left_wrist"], "cam_left_wrist"),
                    "cam_right_wrist": self._to_hwc_rgb(images["cam_right_wrist"], "cam_right_wrist"),
                },
            },
            "apply_gripper_postprocess": bool(obs.get("apply_gripper_postprocess", False)),
            "apply_real_gripper_snap": bool(obs.get("apply_real_gripper_snap", True)),
        }
        for key in (
            "action_horizon",
            "num_inference_steps",
            "sigma_shift",
            "seed",
            "text_cfg_scale",
            "negative_prompt",
            "rand_device",
            "tiled",
            "binarize_gripper",
        ):
            if key in obs:
                payload[key] = obs[key]

        result = self.service.infer_action_chunk(payload)
        action_chunk = np.asarray(result["action_chunk"], dtype=np.float32)
        return {
            "actions": action_chunk,
            "action_chunk": action_chunk,
            "latency_s": float(result["latency_s"]),
            "action_horizon": int(result["action_horizon"]),
            "task_description": str(result["task_description"]),
            "apply_gripper_postprocess": bool(result["apply_gripper_postprocess"]),
            "apply_real_gripper_snap": bool(result["apply_real_gripper_snap"]),
        }

    def reset(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a real-robot Light-WAM policy.")
    parser.add_argument("--ckpt", required=True, help="Path to a Light-WAM checkpoint.")
    parser.add_argument("--config-path", default=None, help="Optional path to resolved training config.yaml.")
    parser.add_argument(
        "--dataset-stats-path",
        default=None,
        help="Optional path to dataset_stats.json. Auto-resolved from the checkpoint directory when omitted.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind.")
    parser.add_argument("--port", type=int, default=5566, help="Port to listen on.")
    parser.add_argument(
        "--protocol",
        choices=("openpi", "raw"),
        default="openpi",
        help="openpi = websocket/msgpack for openpi-client; raw = legacy length-prefixed JSON socket.",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--mixed-precision", default=None, help="Optional override: no / fp16 / bf16.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Optional action horizon override.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Optional denoising step override.")
    parser.add_argument("--sigma-shift", type=float, default=None, help="Optional scheduler sigma-shift override.")
    parser.add_argument("--text-cfg-scale", type=float, default=1.0, help="Default text CFG scale.")
    parser.add_argument("--negative-prompt", default="", help="Default negative prompt.")
    parser.add_argument("--rand-device", default="cpu", help="Random tensor device used inside inference.")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE encode/decode.")
    parser.add_argument("--no-binarize-gripper", action="store_true", help="Disable gripper binarization.")
    parser.add_argument(
        "--disable-real-gripper-snap",
        action="store_true",
        help="Disable real-robot gripper clip/round/close-snap postprocess on gripper dims.",
    )
    parser.add_argument(
        "--real-gripper-dim-indices",
        default="6,13",
        help="Comma-separated action dimension indices treated as real-robot gripper channels.",
    )
    parser.add_argument(
        "--real-gripper-close-snap-threshold",
        type=float,
        default=5.0,
        help="After denormalization and rounding, values within this distance from the minimum are snapped to the fully-closed minimum.",
    )
    parser.add_argument(
        "--disable-prompt-cache",
        action="store_true",
        help="Disable per-task prompt context caching inside the service.",
    )
    parser.add_argument(
        "--query-save-dir",
        default=None,
        help="Optional directory to save each request's images, task text, state, stitched canvas, and response summary.",
    )
    parser.add_argument(
        "--disable-query-save",
        action="store_true",
        help="Disable per-request local dumps.",
    )
    return parser.parse_args()


def _parse_int_list(text: str) -> tuple[int, ...]:
    if _is_none_like(text):
        return tuple()
    values = [item.strip() for item in str(text).split(",")]
    return tuple(int(item) for item in values if item)


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve_checkpoint(args.ckpt)
    config_path = _resolve_config_path(ckpt_path, args.config_path)
    dataset_stats_path = _resolve_dataset_stats_path(ckpt_path, args.dataset_stats_path)

    service = RealRobotPolicyService(
        ckpt_path=ckpt_path,
        config_path=config_path,
        dataset_stats_path=dataset_stats_path,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        rand_device=args.rand_device,
        tiled=args.tiled,
        binarize_gripper=not args.no_binarize_gripper,
        use_prompt_cache=not args.disable_prompt_cache,
        query_save_dir=args.query_save_dir,
        save_queries=not args.disable_query_save,
        gripper_dim_indices=_parse_int_list(args.real_gripper_dim_indices),
        enable_real_gripper_snap=not args.disable_real_gripper_snap,
        real_gripper_close_snap_threshold=float(args.real_gripper_close_snap_threshold),
    )
    if args.protocol == "openpi":
        policy = LightWAMOpenPIAdapter(service)
        server = WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=service.health(None),
        )
        logger.info("Serving Light-WAM through OpenPI websocket protocol on %s:%d", args.host, args.port)
        server.serve_forever()
        return

    server = PolicySocketServer(service, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down real-robot policy service")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
