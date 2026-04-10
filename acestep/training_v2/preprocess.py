"""
Two-Pass CLI Preprocessing for ACE-Step Training V2.
Modified with AUTOMATIC WAV CONVERSION to bypass torchcodec/NVIDIA bugs!
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

# Split-out helpers
from acestep.training_v2.preprocess_discovery import (
    discover_audio_files as _discover_audio_files,
    load_dataset_metadata as _load_dataset_metadata,
    load_sample_metadata as _load_sample_metadata,
    select_genre_indices as _select_genre_indices,
)
from acestep.training_v2.preprocess_prompt import (
    build_simple_prompt as _build_simple_prompt,
)
from acestep.training_v2.preprocess_vae import (
    TARGET_SR as _TARGET_SR,
    tiled_vae_encode as _tiled_vae_encode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_audio_files(
    audio_dir: Optional[str],
    output_dir: str,
    checkpoint_dir: str,
    variant: str = "turbo",
    max_duration: float = 240.0,
    dataset_json: Optional[str] = None,
    device: str = "auto",
    precision: str = "auto",
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict[str, Any]:
    from acestep.training_v2.gpu_utils import detect_gpu

    gpu = detect_gpu(device, precision)
    dev = gpu.device
    prec = gpu.precision

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    audio_files = _discover_audio_files(audio_dir, dataset_json)
    if not audio_files:
        logger.warning("[Side-Step] No audio files found")
        return {"processed": 0, "failed": 0, "total": 0, "output_dir": str(out_path)}

    total = len(audio_files)
    logger.info("[Side-Step] Found %d audio files to preprocess", total)

    sample_meta = _load_sample_metadata(dataset_json, audio_files)
    ds_meta = _load_dataset_metadata(dataset_json)

    ds_tag = ds_meta.get("custom_tag", "")
    if ds_tag:
        for sm in sample_meta.values():
            if not sm.get("custom_tag"):
                sm["custom_tag"] = ds_tag

    intermediates, pass1_failed = _pass1_light(
        audio_files=audio_files,
        sample_meta=sample_meta,
        ds_meta=ds_meta,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        max_duration=max_duration,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    processed, pass2_failed = _pass2_heavy(
        intermediates=intermediates,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    failed = pass1_failed + pass2_failed
    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "output_dir": str(out_path),
    }
    logger.info(
        "[Side-Step] Preprocessing complete: %d/%d processed, %d failed",
        processed,
        total,
        failed,
    )
    return result


# ---------------------------------------------------------------------------
# Pass 1 -- Light models (VAE + Text Encoder)
# ---------------------------------------------------------------------------

def _pass1_light(
    audio_files: List[Path],
    sample_meta: Dict[str, Dict[str, Any]],
    ds_meta: Dict[str, Any],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    max_duration: float,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[List[Path], int]:
    
    from acestep.training_v2.model_loader import (
        load_vae,
        load_text_encoder,
        load_silence_latent,
        unload_models,
        _resolve_dtype,
    )
    from acestep.training.dataset_builder_modules.preprocess_audio import (
        load_audio_stereo,
    )
    from acestep.training.dataset_builder_modules.preprocess_text import encode_text
    from acestep.training.dataset_builder_modules.preprocess_lyrics import encode_lyrics

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 1/2: Loading VAE + Text Encoder ...")
    vae = load_vae(checkpoint_dir, device, precision)
    tokenizer, text_enc = load_text_encoder(checkpoint_dir, device, precision)
    silence_latent = load_silence_latent(
        checkpoint_dir, device, precision, variant=variant
    )

    intermediates: List[Path] = []
    failed = 0
    total = len(audio_files)

    tag_position = ds_meta.get("tag_position", "prepend")
    genre_ratio = ds_meta.get("genre_ratio", 0)
    genre_indices = _select_genre_indices(total, genre_ratio)

    try:
        for i, af in enumerate(audio_files):
            if cancel_check and cancel_check():
                break

            if progress_callback:
                progress_callback(i, total, f"[Pass 1] {af.name}")

            final_pt = out_path / f"{af.stem}.pt"
            if final_pt.exists():
                logger.info("[Side-Step] Skipping (final exists): %s", af.name)
                continue

            try:
                # ====================================================================
                # 🚀 מעקף אוטומטי ל-WAV: מונע קריסות של NVIDIA torchcodec!
                # ====================================================================
                if af.suffix.lower() != '.wav':
                    wav_path = af.with_suffix('.wav')
                    if not wav_path.exists():
                        logger.info(f"[Side-Step] Auto-converting {af.name} to WAV...")
                        os.system(f'ffmpeg -y -i "{af}" "{wav_path}" -loglevel error')
                    af = wav_path  # התוכנה עכשיו חושבת שזה היה קובץ WAV מההתחלה
                # ====================================================================

                # 1. Load audio (stereo, 48 kHz) - עכשיו זה בטוח יעבוד!
                audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)
                audio = audio.unsqueeze(0).to(device=device, dtype=vae.dtype)

                with torch.no_grad():
                    target_latents = _tiled_vae_encode(vae, audio, dtype)

                del audio

                latent_length = target_latents.shape[1]
                attention_mask = torch.ones(
                    1, latent_length, device=device, dtype=dtype
                )

                sm = sample_meta.get(af.name, {})
                caption = sm.get("caption", af.stem)
                lyrics = sm.get("lyrics", "[Instrumental]")

                use_genre = i in genre_indices
                text_prompt = _build_simple_prompt(
                    sm, tag_position=tag_position, use_genre=use_genre
                )

                with torch.no_grad():
                    text_hs, text_mask = encode_text(
                        text_enc, tokenizer, text_prompt, device, dtype
                    )
                    lyric_hs, lyric_mask = encode_lyrics(
                        text_enc, tokenizer, lyrics, device, dtype
                    )

                tmp_path = out_path / f"{af.stem}.tmp.pt"
                torch.save(
                    {
                        "target_latents": target_latents.squeeze(0).cpu(),
                        "attention_mask": attention_mask.squeeze(0).cpu(),
                        "text_hidden_states": text_hs.cpu(),
                        "text_attention_mask": text_mask.cpu(),
                        "lyric_hidden_states": lyric_hs.cpu(),
                        "lyric_attention_mask": lyric_mask.cpu(),
                        "silence_latent": silence_latent.cpu(),
                        "latent_length": latent_length,
                        "metadata": {
                            "audio_path": str(af),
                            "filename": af.name,
                            "caption": caption,
                            "lyrics": lyrics,
                        },
                    },
                    tmp_path,
                )

                del target_latents, attention_mask, text_hs, text_mask
                del lyric_hs, lyric_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                intermediates.append(tmp_path)
                logger.info("[Side-Step] Pass 1 OK: %s", af.name)

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 1 FAIL %s: %s", af.name, exc)

    finally:
        logger.info("[Side-Step] Unloading VAE + Text Encoder ...")
        unload_models(vae, text_enc, tokenizer, silence_latent)

    return intermediates, failed


# ---------------------------------------------------------------------------
# Pass 2 -- Heavy model (DIT encoder)
# ---------------------------------------------------------------------------

def _pass2_heavy(
    intermediates: List[Path],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[int, int]:
    if not intermediates:
        return 0, 0

    from acestep.training_v2.model_loader import (
        load_decoder_for_training,
        unload_models,
        _resolve_dtype,
    )
    from acestep.training.dataset_builder_modules.preprocess_encoder import run_encoder
    from acestep.training.dataset_builder_modules.preprocess_context import (
        build_context_latents,
    )

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 2/2: Loading DIT model (variant=%s) ...", variant)
    model = load_decoder_for_training(checkpoint_dir, variant, device, precision)

    processed = 0
    failed = 0
    total = len(intermediates)

    try:
        for i, tmp_path in enumerate(intermediates):
            if cancel_check and cancel_check():
                break

            if progress_callback:
                progress_callback(i, total, f"[Pass 2] {tmp_path.stem}")

            try:
                data = torch.load(str(tmp_path), weights_only=True)

                model_device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype

                text_hs = data["text_hidden_states"].to(model_device, dtype=model_dtype)
                text_mask = data["text_attention_mask"].to(model_device, dtype=model_dtype)
                lyric_hs = data["lyric_hidden_states"].to(model_device, dtype=model_dtype)
                lyric_mask = data["lyric_attention_mask"].to(model_device, dtype=model_dtype)
                silence_latent = data["silence_latent"].to(model_device, dtype=model_dtype)
                latent_length = data["latent_length"]

                encoder_hs, encoder_mask = run_encoder(
                    model,
                    text_hidden_states=text_hs,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hs,
                    lyric_attention_mask=lyric_mask,
                    device=str(model_device),
                    dtype=model_dtype,
                )

                del text_hs, text_mask, lyric_hs, lyric_mask

                if silence_latent.dim() == 2:
                    silence_latent = silence_latent.unsqueeze(0)

                context_latents = build_context_latents(
                    silence_latent,
                    latent_length,
                    str(model_device),
                    model_dtype,
                )
                del silence_latent

                base_name = tmp_path.name.replace(".tmp.pt", ".pt")
                final_path = out_path / base_name
                
                # תוספת קטנה: תיקון אוטומטי של הממדים השבורים משלב האימון הקודם!
                t_latents = data["target_latents"]
                if t_latents.dim() == 2:
                    t_latents = t_latents.unsqueeze(0)
                    
                torch.save(
                    {
                        "target_latents": t_latents,
                        "attention_mask": data["attention_mask"],
                        "encoder_hidden_states": encoder_hs.squeeze(0).cpu(),
                        "encoder_attention_mask": encoder_mask.squeeze(0).cpu(),
                        "context_latents": context_latents.squeeze(0).cpu(),
                        "metadata": data["metadata"],
                    },
                    final_path,
                )

                del encoder_hs, encoder_mask, context_latents, data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                tmp_path.unlink(missing_ok=True)
                processed += 1
                logger.info("[Side-Step] Pass 2 OK: %s", tmp_path.stem)

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 2 FAIL %s: %s", tmp_path.stem, exc)

    finally:
        logger.info("[Side-Step] Unloading DIT model ...")
        unload_models(model)

    return processed, failed
