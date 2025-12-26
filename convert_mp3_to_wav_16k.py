import os
import librosa
import soundfile as sf

# ==================================
# YOUR ACTUAL AUDIO PATH
# ==================================
BASE_DIR = r"A:\fyp_project\Speech-to-Text-Predictor-main"
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw", "audio")

TARGET_SR = 16000

def convert_mp3_to_wav():
    mp3_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print("❌ No MP3 files found in data/raw")
        return

    print(f"Found {len(mp3_files)} MP3 file(s). Converting to 16kHz WAV...")

    for mp3 in mp3_files:
        input_path = os.path.join(AUDIO_DIR, mp3)
        output_name = os.path.splitext(mp3)[0] + ".wav"
        output_path = os.path.join(AUDIO_DIR, output_name)

        try:
            # Load MP3 → mono → resample to 16kHz
            audio, _ = librosa.load(input_path, sr=TARGET_SR, mono=True)

            if len(audio) == 0:
                print(f"⚠️ Skipped empty file: {mp3}")
                continue

            # Save WAV
            sf.write(output_path, audio, TARGET_SR)
            print(f"✅ Converted: {mp3} → {output_name}")

        except Exception as e:
            print(f"❌ Failed to convert {mp3}: {e}")

    print("\n🎉 Conversion complete.")

if __name__ == "__main__":
    convert_mp3_to_wav()
