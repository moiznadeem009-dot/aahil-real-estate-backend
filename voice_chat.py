import os
import re
import subprocess
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


# ==================================================
# SETTINGS
# ==================================================

API_URL = "http://127.0.0.1:8000/chat/1"

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

AUDIO_FILE = Path("voice.wav")
REPLY_AUDIO_FILE = Path("reply.wav")

PIPER_MODEL = "ur_PK-fasih-medium"


# ==================================================
# START
# ==================================================

print("=" * 50)
print("🤖 ENTERPRISE VOICE CHATBOT")
print("=" * 50)

print()
print("Whisper load ho raha hai...")

whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)

print("✅ Whisper ready hai!")

print()
print("Voice chatbot ready hai.")
print("Roman Urdu mein bolo.")
print("Band karne ke liye Ctrl+C press karo.")


# ==================================================
# URDU → ROMAN URDU
# ==================================================

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
        "ع": "",
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
        "ء": "",
        "ی": "y",
        "ے": "e",
        "ئ": "y",
        "۔": ".",
        "،": ",",
    }

    result = ""

    for char in text:
        result += replacements.get(char, char)

    return result


# ==================================================
# SIMPLE ROMAN URDU NORMALIZATION
# ==================================================

def normalize_roman(text):

    text = text.strip()

    replacements = {
        "myra": "mera",
        "myre": "mere",
        "myri": "meri",
        "nam": "naam",
        "aahyl": "Aahil",
        "aahil": "Aahil",
        "he": "hai",
        "hy": "hai",
        "h": "hai",
    }

    words = text.split()

    fixed_words = []

    for word in words:
        lower = word.lower()

        if lower in replacements:
            fixed_words.append(replacements[lower])
        else:
            fixed_words.append(word)

    return " ".join(fixed_words)


# ==================================================
# RECORD VOICE
# ==================================================

def record_voice():

    print()
    print("=" * 50)
    print("🎙️ 5 seconds tak bolo...")
    print("=" * 50)

    try:

        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )

        sd.wait()

        if audio is None or len(audio) == 0:
            print("❌ Recording empty hai.")
            return False

        sf.write(
            str(AUDIO_FILE),
            audio,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        print("✅ Recording complete!")

        return True

    except Exception as error:

        print()
        print("❌ Microphone error:")
        print(error)

        return False


# ==================================================
# SPEECH → TEXT
# ==================================================

def voice_to_text():

    print()
    print("🎧 Voice ko text mein convert kiya ja raha hai...")

    try:

        segments, info = whisper.transcribe(
            str(AUDIO_FILE),
            language="ur",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        urdu_text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if not urdu_text:
            return ""

        roman_text = urdu_to_roman(urdu_text)

        roman_text = normalize_roman(
            roman_text
        )

        return roman_text

    except Exception as error:

        print()
        print("❌ Whisper error:")
        print(error)

        return ""


# ==================================================
# SEND TO CHATBOT
# ==================================================

def send_to_chatbot(user_text):

    print()
    print("🤖 Chatbot ko message bheja ja raha hai...")

    try:

        response = requests.post(
            API_URL,
            json={
                "message": user_text
            },
            timeout=60,
        )

    except requests.RequestException as error:

        print()
        print("❌ FastAPI se connection nahi hua.")
        print(error)

        return None

    if response.status_code != 200:

        print()
        print("❌ Chatbot error:")
        print(response.text)

        return None

    try:

        data = response.json()

    except ValueError:

        print()
        print("❌ Invalid JSON response.")

        return None

    return data.get("reply")


# ==================================================
# CHATBOT REPLY → VOICE
# ==================================================

def speak_reply(reply):

    print()
    print("=" * 50)
    print("🔊 CHATBOT BOL RAHA HAI")
    print("=" * 50)

    print()
    print(reply)
    print()

    try:

        command = [
            "piper",
            "--model",
            PIPER_MODEL,
            "--length_scale",
            "1.05",
            "--output_file",
            str(REPLY_AUDIO_FILE),
        ]

        process = subprocess.run(
            command,
            input=reply.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if process.returncode != 0:

            print("❌ Piper TTS error:")

            error_text = process.stderr.decode(
                "utf-8",
                errors="replace",
            )

            print(error_text)

            return False

        if not REPLY_AUDIO_FILE.exists():

            print("❌ Reply audio create nahi hui.")

            return False

        print("✅ Voice generate ho gayi.")

        return True

    except FileNotFoundError:

        print("❌ Piper command nahi mila.")

        return False

    except Exception as error:

        print("❌ Voice generation error:")
        print(error)

        return False


# ==================================================
# PLAY REPLY
# ==================================================

def play_reply():

    try:

        os.startfile(
            str(REPLY_AUDIO_FILE)
        )

        return True

    except Exception as error:

        print()
        print("❌ Audio play nahi hui.")
        print(error)

        return False


# ==================================================
# MAIN LOOP
# ==================================================

def main():

    while True:

        try:

            if not record_voice():
                continue

            user_text = voice_to_text()

            print()
            print("=" * 50)
            print("📝 TUMNE KAHA:")
            print("=" * 50)

            if not user_text:

                print("(kuch detect nahi hua)")
                continue

            print(user_text)

            print("=" * 50)

            reply = send_to_chatbot(
                user_text
            )

            if not reply:
                continue

            print()
            print("=" * 50)
            print("🤖 CHATBOT:")
            print("=" * 50)

            print(reply)

            print("=" * 50)

            success = speak_reply(
                reply
            )

            if success:
                play_reply()

            print()
            print("-" * 50)
            print("🎙️ Dobara Roman Urdu mein bolo...")
            print("-" * 50)

        except KeyboardInterrupt:

            print()
            print("👋 Voice chatbot band ho gaya.")

            break

        except Exception as error:

            print()
            print("❌ Unexpected error:")
            print(error)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()