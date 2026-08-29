from faster_whisper import WhisperModel
from pathlib import Path

AUDIO_FILE = Path("voice.wav")

if not AUDIO_FILE.exists():
    raise FileNotFoundError(
        "voice.wav nahi mila. Pehle python mic_record.py chalao."
    )

print("Local Whisper model load ho raha hai...")
print("Pehli baar model download hoga, ismein time lag sakta hai.")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)

print("Voice ko text mein convert kiya ja raha hai...")

segments, info = model.transcribe(
    str(AUDIO_FILE),
    language="en",
)

print()
print("================================")
print("TRANSCRIPTION:")
print("================================")

full_text = ""

for segment in segments:
    full_text += segment.text

print(full_text.strip())

print("================================")
print("Voice → Text complete!")