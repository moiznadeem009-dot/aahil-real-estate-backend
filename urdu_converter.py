
import subprocess
import os
import time

MODEL = "ur_PK-fasih-medium"
OUTPUT_FILE = "chatbot_voice.wav"


def speak(text):
    """Roman Urdu text ko Piper Fasih voice mein convert karta hai."""

    print("\n🔊 Chatbot bol raha hai...")
    print("🤖:", text)

    # Purani audio file delete karo
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Piper ko text do
    process = subprocess.run(
        [
            "piper",
            "--model", MODEL,
            "--length_scale", "1.0",
            "--output_file", OUTPUT_FILE
        ],
        input=text,
        text=True,
        encoding="utf-8",
        capture_output=True
    )

    if process.returncode != 0:
        print("❌ Piper error:")
        print(process.stderr)
        return

    # Audio play karo
    os.startfile(OUTPUT_FILE)


def main():
    print("=" * 40)
    print("🤖 Roman Urdu Voice Chatbot")
    print("=" * 40)
    print("Roman Urdu mein message likho.")
    print("Band karne ke liye 'exit' likho.\n")

    while True:
        text = input("📝 Tum: ").strip()

        if text.lower() == "exit":
            print("👋 Chatbot band ho gaya.")
            break

        if not text:
            continue

        # Filhaal test response
        reply = f"Theek hai Aahil, tumne kaha: {text}"

        speak(reply)


if __name__ == "__main__":
    main()

