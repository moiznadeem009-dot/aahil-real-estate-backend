import sounddevice as sd
import wave

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5
OUTPUT_FILE = "voice.wav"

print("Recording start...")
print("5 seconds tak bolo!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
)

sd.wait()

print("Recording complete!")

with wave.open(OUTPUT_FILE, "wb") as wav_file:
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.writeframes(audio.tobytes())

print(f"Audio save ho gayi: {OUTPUT_FILE}")