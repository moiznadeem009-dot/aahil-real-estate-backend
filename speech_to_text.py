import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "voice.wav"


print("Whisper load ho raha hai...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Whisper ready!")


def urdu_to_roman(text):
    replacements = {
        "آ": "aa",
        "ا": "a",
        "ب": "b",
        "پ": "p",
        "ت": "t",
        "ٹ": "t",
        "ث": "s",
        "ج": "j",
        "چ": "ch",
        "ح": "h",
        "خ": "kh",
        "د": "d",
        "ڈ": "d",
        "ذ": "z",
        "ر": "r",
        "ڑ": "r",
        "ز": "z",
        "ژ": "zh",
        "س": "s",
        "ش": "sh",
        "ص": "s",
        "ض": "z",
        "ط": "t",
        "ظ": "z",
        "ع": "'",
        "غ": "gh",
        "ف": "f",
        "ق": "q",
        "ک": "k",
        "گ": "g",
        "ل": "l",
        "م": "m",
        "ن": "n",
        "ں": "n",
        "و": "w",
        "ہ": "h",
        "ھ": "h",
        "ء": "'",
        "ی": "y",
        "ے": "e",
        "ئ": "y",
        "۔": ".",
        "،": ",",
    }

    return "".join(
        replacements.get(char, char)
        for char in text
    )


def main():

    print()
    print("5 seconds tak Roman Urdu mein bolo...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        AUDIO_FILE,
        audio,
        SAMPLE_RATE
    )

    print("Recording complete.")
    print("Voice ko text mein convert kar raha hoon...")

    segments, info = model.transcribe(
        AUDIO_FILE,
        language="ur",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    urdu_text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    roman_text = urdu_to_roman(urdu_text)

    print()
    print("=" * 40)
    print("URDU TEXT:")
    print("=" * 40)
    print(urdu_text)

    print()
    print("=" * 40)
    print("ROMAN URDU:")
    print("=" * 40)
    print(roman_text)
    print("=" * 40)


if __name__ == "__main__":
    main()