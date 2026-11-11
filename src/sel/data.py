"""Native Qwen input packing and source-coordinate alignment."""

import gzip
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCHEMA_VERSION = 3


class SkipExample(ValueError):
    """Explicit exclusion, never truncation or silent modality removal."""


def has_audio(record):
    suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
    return any(
        m.get("kind") == "audio" or Path(m.get("path", "")).suffix.lower() in suffixes
        for m in record.get("media", [])
    ) or any(
        Path(s.get("asset", "")).suffix.lower() in suffixes
        for a in record["solutions"]
        for sels in a.values()
        for s in sels
    )


def all_selections(record):
    return [s for a in record["solutions"] for sels in a.values() for s in sels]


def _video_frames(path, max_fps=8, max_frames=72):
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
    )
    stream = probe["streams"][0]
    w, h = stream["width"], stream["height"]
    rate = stream["avg_frame_rate"] or stream["r_frame_rate"]
    a, b = map(int, rate.split("/"))
    fps = a / b
    timestamps = np.array(
        [
            float(f["best_effort_timestamp_time"])
            for f in probe.get("frames", [])
            if "best_effort_timestamp_time" in f
        ],
        dtype=float,
    )
    if not len(timestamps):
        raise SkipExample("video has no exact frame timestamps")
    timestamps -= timestamps[0]
    duration = float(stream.get("duration") or (timestamps[-1] + 1 / fps))
    n = min(
        len(timestamps), max_frames, max(2, math.ceil(duration * min(fps, max_fps)))
    )
    desired = np.arange(n) * duration / n
    indices = np.searchsorted(timestamps, desired).clip(0, len(timestamps) - 1)
    indices = np.unique(indices).tolist()
    expression = "+".join(f"eq(n\\,{i})" for i in indices)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
            str(path),
            "-vf",
            f"select={expression}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        timeout=40,
        check=True,
    )
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w, 3).copy()
    if len(frames) != len(indices):
        raise SkipExample("native decoded video frame count mismatch")
    pts = timestamps[indices].tolist()
    return frames, {
        "asset": path.name,
        "fps": fps,
        "duration": duration,
        "width": w,
        "height": h,
        "total_num_frames": len(timestamps),
        "frame_indices": indices,
        "timestamps": pts,
    }


def _overlap(a, b, c, d):
    return a < d and b > c


def _clip_polygon(points, bounds):
    x0, y0, x1, y1 = bounds
    poly = [tuple(p) for p in points]
    for axis, value, sign in [(0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)]:
        result = []
        if not poly:
            break
        for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
            ia = sign * (a[axis] - value) >= 0
            ib = sign * (b[axis] - value) >= 0
            if ia:
                result.append(a)
            if ia != ib:
                t = (value - a[axis]) / (b[axis] - a[axis])
                result.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(2)))
        poly = result
    return poly


def _spatial_hit(selection, rect):
    x0, y0, x1, y1 = rect
    kind = selection["kind"]
    if kind == "box":
        a, b, c, d = selection["xyxy"]
        return _overlap(x0, x1, a, c) and _overlap(y0, y1, b, d)
    if kind == "point":
        x, y = selection["xy"]
        radius = selection.get("radius", 1)
        return (
            (x - min(max(x, x0), x1)) ** 2 + (y - min(max(y, y0), y1)) ** 2 < radius**2
            or x0 <= x < x1
            and y0 <= y < y1
        )
    if kind == "polygon":
        poly = _clip_polygon(selection.get("points", selection.get("xy", [])), rect)
        return (
            len(poly) >= 3
            and abs(
                sum(
                    a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(poly, poly[1:] + poly[:1], strict=True)
                )
            )
            > 1e-8
        )
    return False


def _unit_matches(selection, unit):
    kind = selection["kind"]
    if kind == "span":
        return unit["kind"] == "text" and _overlap(
            unit["start"], unit["end"], selection["start"], selection["end"]
        )
    if kind == "time":
        return (
            unit["kind"] == "video"
            and unit["asset"] == selection.get("asset", "source.mp4")
            and _overlap(
                unit["start"], unit["end"], selection["start"], selection["end"]
            )
        )
    # Spatial labels without an asset refer to the rendered source image. Time
    # labels may carry xyxy gallery overlays; they are NOT spatial supervision.
    return (
        unit["kind"] == "image"
        and unit["asset"] == selection.get("asset", unit["asset"])
        and _spatial_hit(selection, unit["xyxy"])
    )


def _queries(queries):
    if not isinstance(queries, (list, tuple)) or not queries:
        raise ValueError("Provide at least one literal selection query")
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("Queries must be nonempty strings")
    return [{"id": f"q{i + 1}", "text": q} for i, q in enumerate(queries)]


def pack_sources(
    processor,
    *,
    text=None,
    image_path=None,
    video_paths=(),
    queries,
    max_length=8192,
    image_asset=None,
    video_assets=None,
    query_first=False,
):
    """Pack actual source surfaces without constructing or consulting labels."""
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer")
    if text is not None and not isinstance(text, str):
        raise TypeError("Source text must be a string")
    source_text = text
    use_text = bool(source_text)
    use_image = image_path is not None
    image_path = Path(image_path).resolve() if use_image else None
    directory = Path("/")
    video_paths = [Path(path).resolve() for path in video_paths]
    video_assets = video_assets or [path.name for path in video_paths]
    if len(video_assets) != len(video_paths):
        raise ValueError("Every video needs its own literal asset name")
    video_media = [
        {"kind": "video", "path": str(path), "asset": asset}
        for path, asset in zip(video_paths, video_assets, strict=True)
    ]
    record = {
        "source_image": str(image_asset or (image_path.name if image_path else "")),
        "queries": _queries(queries),
    }
    if not (use_text or use_image or video_media):
        raise ValueError("Provide text, an image or a video source")
    images = []
    videos = []
    video_info = []
    content = []
    if use_image:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        content.extend(
            [
                {
                    "type": "text",
                    "text": "Source image (rules, objects and selectable surface):\n",
                },
                {"type": "image"},
            ]
        )
    if use_text:
        content.append(
            {
                "type": "text",
                "text": "\nSource text:\n" + source_text + "\nEnd source text.\n",
            }
        )
    for media in video_media:
        frames, info = _video_frames(directory / media["path"])
        videos.append(frames)
        video_info.append(info)
        content.extend(
            [
                {"type": "text", "text": "\nVideo asset: " + media["asset"] + "\n"},
                {"type": "video"},
            ]
        )
    queries = record["queries"]
    query_lines = [f"Question {i + 1}: {q['text']}" for i, q in enumerate(queries)]
    # Chat templates trim the final text block. A literal sentinel is needed
    # only when the native last question ends in whitespace; do not strip it.
    trailing_guard = bool(query_lines[-1] != query_lines[-1].rstrip())
    ending = "\nEnd selection questions.\n" if trailing_guard else "\n"
    query_block = {
        "type": "text",
        "text": "\nSelection questions:\n" + "\n".join(query_lines) + ending,
    }
    if query_first:
        content.insert(0, query_block)
    else:
        content.append(query_block)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    kwargs = {
        "text": prompt,
        "return_tensors": "pt",
        "add_special_tokens": False,
        "padding": False,
        "truncation": False,
    }
    if images:
        kwargs["images"] = images
        kwargs["images_kwargs"] = {
            "size": {
                "shortest_edge": 65536,
                "longest_edge": (2000 if videos else 6000) * 32 * 32,
            }
        }
    if videos:
        from transformers.video_utils import VideoMetadata

        kwargs["videos"] = videos
        # Two native frames form each temporal patch. Budget counts merged tokens.
        budget = 4000 / len(videos)
        kwargs["videos_kwargs"] = {
            "do_sample_frames": False,
            "size": {"shortest_edge": 4096, "longest_edge": int(budget * 32 * 32 * 2)},
            "cap_pixels_per_frame": False,
            "video_metadata": [
                VideoMetadata(
                    total_num_frames=i["total_num_frames"],
                    fps=i["fps"],
                    width=i["width"],
                    height=i["height"],
                    duration=i["duration"],
                    frames_indices=[t * i["fps"] for t in i["timestamps"]],
                )
                for i in video_info
            ],
        }
    encoded = processor(**kwargs)
    inputs = {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    ids = inputs["input_ids"][0].tolist()
    if len(ids) > max_length:
        raise SkipExample(
            f"oversize:{len(ids)}>{max_length}; no source truncation/windowing"
        )
    expanded = processor.tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    roundtrip = processor.tokenizer(
        expanded, add_special_tokens=False, return_offsets_mapping=True
    )
    if roundtrip["input_ids"] != ids:
        raise SkipExample("expanded_prompt_token_roundtrip_mismatch")
    offsets = roundtrip["offset_mapping"]
    q_positions = []
    header = "Selection questions:\n"
    if query_first:
        search = expanded.find(header)
    else:
        header = "\n" + header
        search = expanded.rfind(header)
    if search < 0:
        raise SkipExample("selection_question_header_missing")
    search += len(header)
    for line in query_lines:
        a = expanded.find(line, search)
        if a < 0:
            raise SkipExample("literal_query_not_preserved")
        b = a + len(line)
        positions = [i for i, (x, y) in enumerate(offsets) if _overlap(x, y, a, b)]
        if not positions:
            raise SkipExample("query_has_no_tokens")
        q_positions.append(positions[-1])
        search = b
    source_indices = []
    units = []
    if use_text:
        marker = "\nSource text:\n"
        a = expanded.find(marker, search if query_first else 0)
        if a < 0:
            raise SkipExample("source_marker_missing")
        a += len(marker)
        if expanded[a : a + len(source_text)] != source_text:
            raise SkipExample("raw_source_not_preserved_after_processor")
        for i, (x, y) in enumerate(offsets):
            if _overlap(x, y, a, a + len(source_text)):
                source_indices.append(i)
                units.append(
                    {
                        "kind": "text",
                        "start": max(x - a, 0),
                        "end": min(y - a, len(source_text)),
                    }
                )
    image_id = processor.image_token_id
    video_id = processor.video_token_id
    image_positions = [i for i, v in enumerate(ids) if v == image_id]
    if images:
        t, h, w = map(int, inputs["image_grid_thw"][0])
        hh, ww = h // 2, w // 2
        if len(image_positions) != t * hh * ww:
            raise SkipExample("expanded_image_grid_token_mismatch")
        ow, oh = images[0].size
        for k, pos in enumerate(image_positions):
            r, c = divmod(k, ww)
            source_indices.append(pos)
            units.append(
                {
                    "kind": "image",
                    "asset": record["source_image"],
                    "xyxy": [
                        c * ow / ww,
                        r * oh / hh,
                        (c + 1) * ow / ww,
                        (r + 1) * oh / hh,
                    ],
                    "grid": [r, c, hh, ww],
                    "original_size": [ow, oh],
                    "resized_size": [w * 16, h * 16],
                }
            )
    video_positions = [i for i, v in enumerate(ids) if v == video_id]
    cursor = 0
    for info, grid in zip(video_info, inputs.get("video_grid_thw", []), strict=True):
        t, h, w = map(int, grid)
        hh, ww = h // 2, w // 2
        count = t * hh * ww
        positions = video_positions[cursor : cursor + count]
        cursor += count
        if len(positions) != count:
            raise SkipExample("expanded_video_grid_token_mismatch")
        pts = list(info["timestamps"])
        frame_indices = list(info["frame_indices"])
        while len(pts) < t * 2:
            pts.append(pts[-1])
            frame_indices.append(frame_indices[-1])
        if len(pts) != t * 2:
            raise SkipExample("temporal_patch_frame_count_mismatch")
        # Source time bins are Voronoi intervals of each temporal pair's mean PTS.
        centers = [(pts[j * 2] + pts[j * 2 + 1]) / 2 for j in range(t)]
        boundaries = (
            [0]
            + [(a + b) / 2 for a, b in zip(centers, centers[1:], strict=False)]
            + [info["duration"]]
        )
        for k, pos in enumerate(positions):
            temporal, spatial = divmod(k, hh * ww)
            r, c = divmod(spatial, ww)
            source_indices.append(pos)
            units.append(
                {
                    "kind": "video",
                    "asset": info["asset"],
                    "start": boundaries[temporal],
                    "end": boundaries[temporal + 1],
                    "timestamps": pts[temporal * 2 : temporal * 2 + 2],
                    "frame_indices": frame_indices[temporal * 2 : temporal * 2 + 2],
                    "xyxy": [
                        c * info["width"] / ww,
                        r * info["height"] / hh,
                        (c + 1) * info["width"] / ww,
                        (r + 1) * info["height"] / hh,
                    ],
                    "grid": [temporal, r, c, t, hh, ww],
                    "original_size": [info["width"], info["height"]],
                }
            )
    if cursor != len(video_positions):
        raise SkipExample("unused_expanded_video_tokens")
    if len(image_positions) + len(video_positions) > 6000:
        raise SkipExample("visual_source_budget_exceeded")
    order = sorted(range(len(source_indices)), key=source_indices.__getitem__)
    source_indices = [source_indices[i] for i in order]
    units = [units[i] for i in order]
    if not units:
        raise SkipExample("empty_source_units")
    if len(set(source_indices)) != len(source_indices):
        raise SkipExample("duplicate_source_position")
    return {
        "inputs": inputs,
        "source_indices": torch.tensor(source_indices, dtype=torch.long),
        "question_positions": torch.tensor(q_positions, dtype=torch.long),
        "metadata": {
            "source_units": units,
            "source_text": source_text if use_text else None,
            "source_image": str(image_path) if image_path else None,
            "video_metadata": video_info,
            "video_paths": [str(path) for path in video_paths],
            "queries": record["queries"],
            "prompt": prompt,
            "expanded_prompt": expanded,
            "sequence_length": len(ids),
            "visual_tokens": len(image_positions) + len(video_positions),
            "question_whitespace_guard": trailing_guard,
            "modality": "video" if videos else "image" if images else "text",
        },
    }


def _declared_sources(record, directory, input_modality=None):
    media = record.get("media", [])
    if any(m.get("kind") == "audio" for m in media):
        raise SkipExample("Audio inference is not supported in this experiment")
    videos = [directory / m["path"] for m in media if m.get("kind") == "video"]
    surface = input_modality or record.get("input_modality") or record.get("modality")
    text = record.get("source_text")
    # Text/code records' PNGs are gallery renderings, not additional model input.
    # Video always retains its source storyboard/rules image, as in training.
    image = record.get("source_image")
    include_image = bool(image) and (
        bool(videos) or surface in {"image", "mixed"} or not text
    )
    if surface in {"image", "video"}:
        text = None
    if surface in {"text", "code"} and not videos:
        include_image = False
    return {
        "text": text,
        "image_path": directory / image if include_image else None,
        "image_asset": image if include_image else None,
        "video_paths": videos,
        "video_assets": [m["path"] for m in media if m.get("kind") == "video"],
    }


def pack_example(
    path,
    processor,
    *,
    queries=None,
    max_length=8192,
    input_modality=None,
    query_first=False,
):
    """Use only declared source properties; fresh unlabelled records are valid."""
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    sources = _declared_sources(record, path.parent, input_modality)
    literal_queries = (
        queries if queries is not None else [q["text"] for q in record["queries"]]
    )
    packed = pack_sources(
        processor,
        **sources,
        queries=literal_queries,
        max_length=max_length,
        query_first=query_first,
    )
    packed["metadata"]["record_path"] = str(path)
    return packed


def load_example(path, processor, max_length=8192, *, query_first=False):
    """Pack the declared sources, then align supervision to the packed source units."""
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if has_audio(record):
        raise SkipExample("audio_dependency")
    if record.get("verification", {}).get("status") != "verified":
        raise SkipExample("unverified_ground_truth")
    kinds = {s["kind"] for s in all_selections(record)}
    unknown = kinds - {"span", "box", "point", "polygon", "time"}
    if unknown:
        raise SkipExample(f"unsupported_selection_kinds:{sorted(unknown)}")
    queries = record["queries"]
    packed = pack_sources(
        processor,
        **_declared_sources(record, path.parent),
        queries=[query["text"] for query in queries],
        max_length=max_length,
        query_first=query_first,
    )
    return label_packed_example(record, path, packed, query_first=query_first)


def label_packed_example(record, path, packed, *, query_first):
    """Attach whole-answer labels to already packed native source positions."""
    queries = record["queries"]
    units = packed["metadata"]["source_units"]
    targets = torch.zeros(
        (len(record["solutions"]), len(queries), len(units)), dtype=torch.float32
    )
    image_positions = [i for i, unit in enumerate(units) if unit["kind"] == "image"]
    image_rects = np.asarray([units[i]["xyxy"] for i in image_positions])
    image_asset = units[image_positions[0]]["asset"] if image_positions else None
    for a, solution in enumerate(record["solutions"]):
        for q, query in enumerate(queries):
            selections = solution[query["id"]]
            if (
                image_positions
                and len(selections) > 1
                and all(
                    s["kind"] == "box" and s.get("asset", image_asset) == image_asset
                    for s in selections
                )
            ):
                boxes = np.asarray([s["xyxy"] for s in selections])
                matches = (
                    (image_rects[:, None, 0] < boxes[None, :, 2])
                    & (boxes[None, :, 0] < image_rects[:, None, 2])
                    & (image_rects[:, None, 1] < boxes[None, :, 3])
                    & (boxes[None, :, 1] < image_rects[:, None, 3])
                )
                if not matches.any(axis=0).all():
                    raise SkipExample("positive_selection_has_no_aligned_source_unit")
                targets[a, q, image_positions] = torch.from_numpy(
                    matches.any(axis=1)
                ).float()
                continue
            for s in selections:
                matched = [i for i, u in enumerate(units) if _unit_matches(s, u)]
                if not matched:
                    raise SkipExample("positive_selection_has_no_aligned_source_unit")
                targets[a, q, matched] = 1
    packed["metadata"].update(
        schema_version=SCHEMA_VERSION,
        artifact_path=str(path),
        task_id=record["id"],
        instance_index=record.get("instance_index"),
        raw_record=record,
        question_count=len(queries),
        alternative_count=len(record["solutions"]),
        queries=queries,
        query_first=query_first,
    )
    packed["targets"] = targets
    packed["target_mask"] = torch.ones((len(queries), len(units)), dtype=torch.bool)
    return packed


def save_cached_example(row, path):
    """Losslessly code pixel floats before gzip; verify every restored value."""
    packed = {**row, "inputs": dict(row["inputs"])}
    stats = {}
    for key, value in row["inputs"].items():
        if key.startswith("pixel_values"):
            if isinstance(value, dict) and "deferred_image" in value:
                stats[key] = "deferred_image"
                continue
            indices = (
                ((value.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            )
            lut = torch.zeros(256, dtype=value.dtype)
            lut[indices.long().flatten()] = value.flatten()
            if torch.equal(lut[indices.long()], value):
                packed["inputs"][key] = {"pixel_codes": indices, "pixel_lut": lut}
                stats[key] = "lossless_uint8_lut"
            else:
                stats[key] = "lossless_float_fallback"
    packed["metadata"] = {**row["metadata"], "pixel_storage": stats}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path.with_suffix(".tmp"), "wb", compresslevel=1) as f:
        torch.save(packed, f)
    path.with_suffix(".tmp").replace(path)
    return path.stat().st_size


def load_cached_example(path, processor):
    with gzip.open(path, "rb") as f:
        row = torch.load(f, map_location="cpu", weights_only=True)
    for key, value in list(row["inputs"].items()):
        if isinstance(value, dict) and "pixel_codes" in value:
            row["inputs"][key] = value["pixel_lut"][value["pixel_codes"].long()]
        elif isinstance(value, dict) and "deferred_image" in value:
            row["inputs"][key] = _load_deferred_image(
                value, row["inputs"]["image_grid_thw"], processor
            )
    return row


def _load_deferred_image(item, expected_grid, processor):
    raw = Path(item["deferred_image"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError(f"Packed image changed: {item['deferred_image']}")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    result = processor.image_processor(
        images=[image],
        return_tensors="pt",
        size={"shortest_edge": 65536, "longest_edge": 6000 * 32 * 32},
    )
    if not torch.equal(result["image_grid_thw"], expected_grid):
        raise ValueError("Deferred image grid changed since native packing")
    return result["pixel_values"]
