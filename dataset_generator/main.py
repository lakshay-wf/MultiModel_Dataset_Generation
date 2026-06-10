import argparse
import json
import time
import torch
from pathlib import Path
from diffusers import AutoPipelineForText2Image

from generate_images import generate_image
from generate_audio import generate_audio

# ── Model init ───────────────────────────────────────────────────────────────
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MASTER_INDEX = BASE_DIR / "master_index.json"
LOGS_DIR     = BASE_DIR / "logs"


def load_log(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_log(path: Path, data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Dataset Generation Pipeline")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Max number of entries to process for quick testing (e.g., --limit 1)"
    )
    return parser.parse_args()

def main() -> None:
    
    args = parse_args()  

    with open(MASTER_INDEX) as f:
        index: dict = json.load(f)

    entries = [entry for entries in index.values() for entry in entries]

    if args.limit is not None:
        print(f"[INFO] Running in test/limit mode. Processing only the first {args.limit} item(s).")
        entries = entries[:args.limit]


    total = len(entries)

    image_log = load_log(LOGS_DIR / "images_logs.json")
    audio_log = load_log(LOGS_DIR / "audios_logs.json")

    t_start = time.perf_counter()
    print(f"[START] Processing {total} samples...\n")

    for i, entry in enumerate(entries, 1):
        entry_id   = entry["id"]
        caption    = entry["caption"]
        image_path = BASE_DIR / entry["image_file"]
        audio_path = BASE_DIR / entry["audio_file"]

        print(f"[{i}/{total}] {entry_id} — {caption}")

        t0 = time.perf_counter()
        image_ok = generate_image(text=caption, output_path=str(image_path), pipe=pipe)
        image_log[entry_id] = image_ok
        save_log(LOGS_DIR / "images_logs.json", image_log)
        image_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        audio_ok = generate_audio(text=caption, output_path=audio_path)
        audio_log[entry_id] = audio_ok
        save_log(LOGS_DIR / "audios_logs.json", audio_log)
        audio_time = time.perf_counter() - t0

        print(f"  image → {'OK' if image_ok else 'FAILED'} ({image_time:.2f}s)  |  audio → {'OK' if audio_ok else 'FAILED'} ({audio_time:.2f}s)")
        print(f"  total: {image_time + audio_time:.2f}s")

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] {total} samples processed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
