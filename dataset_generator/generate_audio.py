import httpx
from pathlib import Path

TTS_URL = "http://127.0.0.1:8000/generate_audio_wav"


def generate_audio(text: str, output_path: str | Path) -> bool:
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", TTS_URL, json={"text": text}) as response:
                response.raise_for_status()
                audio_bytes = b"".join(response.iter_bytes())

        output_path.write_bytes(audio_bytes)
        return True
    except Exception as e:
        print(f"  [audio error] {e}")
        return False
