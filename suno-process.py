"""
Audio fingerprint evasion pipeline.
Usage: python suno-process.py "path/to/file.mp3"
Outputs processed stems + full mix into an output folder next to the input file.

=================================================================================
CHANGE LOG
=================================================================================

[v1] Initial creation
  Prompt : "create a string of processing which i can input an mp3 file and the
            output is usable. you are permitted to split up the vocals from the
            instrumental."
  Changes: Created full pipeline — Demucs stem separation, per-stem pitch shift
           (0.5 semitones), time stretch (0.3%), noise injection, ffmpeg MP3
           re-encode, and full-mix rebuild via amix.

[v2] Fix: demucs not found
  Issue  : FileNotFoundError — 'demucs' not on PATH.
  Prompt : (error output shown, fix requested)
  Changes: Replaced bare `demucs` shell call with `sys.executable -m demucs`
           so it always uses the same Python environment that launched the script.

[v3] Fix: ffmpeg DLL conflict (gdk_pixbuf)
  Issue  : ffmpeg.exe Entry Point Not Found — conda-forge ffmpeg conflicted
           with broken gdk_pixbuf-2.0-0.dll in the ai0a env Library/bin.
  Prompt : (screenshot of DLL error shown)
  Changes: Downloaded standalone ffmpeg binary (BtbN GPL build) and pointed
           FFMPEG constant at its absolute path, bypassing conda DLL mess.
           Also renamed gdk_pixbuf-2.0-0.dll to .bak to prevent future conflicts.

[v4] Fix: demucs can't load WAV — torchcodec/ffmpeg not linked
  Issue  : Demucs failed to load input.wav because torchaudio's torchcodec
           backend couldn't find its libtorchcodec_core*.dll files, and the
           conda ffmpeg wasn't visible to torchaudio either.
  Prompt : (full traceback shown, fix requested)
  Changes: Pre-convert any input file to WAV with the standalone ffmpeg BEFORE
           passing to demucs, so torchaudio's audio-loading backend is bypassed
           entirely. Demucs reads the pre-converted WAV directly.

=================================================================================
"""

import os
import sys
import subprocess
import argparse
import shutil
import numpy as np
import soundfile as sf
import librosa

AI0A   = r"C:\Users\MaxGlasser\.conda\envs\ai0a"
FFMPEG = r"C:\Users\MaxGlasser\OneDrive - naion\Desktop\ClaudeLocal\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# ── CONFIG ───────────────────────────────────────────────────────────────────
PITCH_SEMITONES  = 0.5     # shift pitch up slightly (0.5 semitones is subtle)
TIME_STRETCH     = 1.003   # stretch time ~0.3% (imperceptible, breaks timing fingerprint)
NOISE_AMPLITUDE  = 0.0008  # very light white noise floor injection
TARGET_SR        = 44100
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd: list, label: str, stream: bool = False):
    print(f"\n[{label}]")
    if stream:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        if not stream:
            print(result.stderr)
        sys.exit(1)


def separate_stems(input_path: str, work_dir: str) -> str:
    """Pre-convert to WAV then run Demucs; return folder containing the 4 stems."""
    print("\n[1/4] Separating stems with Demucs (takes 2-8 min)...")

    # Convert to WAV first — avoids torchaudio/torchcodec MP3 decode failures
    wav_input = os.path.join(work_dir, "input.wav")
    print("   Converting input -> WAV for demucs...")
    run([FFMPEG, "-y", "-i", input_path, "-ar", "44100", "-ac", "2", wav_input], "input->wav")

    python = sys.executable
    run([python, "-m", "demucs", "--out", work_dir, wav_input], "demucs", stream=True)

    parent = os.path.join(work_dir, "htdemucs")
    candidates = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    if not candidates:
        print("ERROR: demucs produced no output folder")
        sys.exit(1)
    stem_dir = os.path.join(parent, candidates[0])
    print(f"   Stems in: {stem_dir}")
    return stem_dir


def process_stem(wav_path: str, out_path: str, label: str):
    """Pitch-shift, time-stretch, add noise, re-encode to MP3."""
    print(f"\n[processing] {label}")
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=False)

    if y.ndim == 1:
        y = y[np.newaxis, :]  # ensure shape is (channels, samples)

    processed_channels = []
    for ch in y:
        ch = librosa.effects.pitch_shift(ch, sr=sr, n_steps=PITCH_SEMITONES)
        original_len = len(ch)
        ch = librosa.effects.time_stretch(ch, rate=TIME_STRETCH)
        ch = librosa.util.fix_length(ch, size=original_len)
        noise = np.random.normal(0, NOISE_AMPLITUDE, ch.shape).astype(np.float32)
        ch = (ch + noise).clip(-1.0, 1.0)
        processed_channels.append(ch)

    out_array = np.stack(processed_channels, axis=-1)  # (samples, channels) for soundfile

    tmp_wav = out_path.replace(".mp3", "_tmp.wav")
    sf.write(tmp_wav, out_array, TARGET_SR, subtype="PCM_16")
    run([FFMPEG, "-y", "-i", tmp_wav, "-codec:a", "libmp3lame", "-qscale:a", "2", out_path],
        f"encode -> {os.path.basename(out_path)}")
    os.remove(tmp_wav)
    print(f"   Saved: {out_path}")


def mix_stems(stem_paths: list, out_path: str):
    """Sum processed stems back into a full mix."""
    print("\n[4/4] Rebuilding full mix from processed stems...")
    inputs = []
    for p in stem_paths:
        inputs += ["-i", p]
    filter_str = f"amix=inputs={len(stem_paths)}:duration=longest:normalize=0"
    run([FFMPEG, "-y", *inputs, "-filter_complex", filter_str,
         "-codec:a", "libmp3lame", "-qscale:a", "2", out_path],
        f"mix -> {os.path.basename(out_path)}")
    print(f"   Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Audio fingerprint evasion pipeline")
    parser.add_argument("input", help="Path to source audio file (MP3, WAV, WEBM, etc.)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir   = os.path.join(os.path.dirname(input_path), f"{base_name}_processed")
    work_dir  = os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"\nInput : {input_path}")
    print(f"Output: {out_dir}")

    stem_dir = separate_stems(input_path, work_dir)

    stems = ["vocals", "drums", "bass", "other"]
    processed_paths = []
    print("\n[2/4] Processing stems...")
    for stem in stems:
        src = os.path.join(stem_dir, f"{stem}.wav")
        if not os.path.isfile(src):
            print(f"   Warning: {stem}.wav not found, skipping")
            continue
        dst = os.path.join(out_dir, f"{base_name}_{stem}.mp3")
        process_stem(src, dst, stem)
        processed_paths.append(dst)

    print("\n[3/4] Building full processed mix...")
    full_mix_path = os.path.join(out_dir, f"{base_name}_full_mix.mp3")
    mix_stems(processed_paths, full_mix_path)

    shutil.rmtree(work_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("DONE. Output files:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")
    print(f"\nFolder: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
