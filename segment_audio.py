import pandas as pd
import librosa
import soundfile as sf
import os
import warnings
from tqdm import tqdm

# Suppress librosa/audioread warnings to keep terminal clean
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
AUDIO_DIR = os.path.join('data', 'raw', 'audio')
SEP_LABELS = os.path.join('data', 'raw', 'SEP-28k_labels.csv')
CLIPS_OUTPUT = os.path.join('data', 'processed', 'clips')

# Ensure output directory exists
os.makedirs(CLIPS_OUTPUT, exist_ok=True)

def segment_data():
    print("\n" + "="*40)
    print("   AUDIO SEGMENTATION ENGINE")
    print("="*40)
    
    if not os.path.exists(SEP_LABELS):
        print(f"Error: Label file not found at {SEP_LABELS}")
        return
    
    # Load labels (SEP-28k uses a specific header format)
    try:
        df = pd.read_csv(SEP_LABELS)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Get list of downloaded episodes physically present on the F: drive
    downloaded_files = [f for f in os.listdir(AUDIO_DIR) if f.startswith("SEP28k")]
    
    if not downloaded_files:
        print("No audio files found in data/raw/audio. Run downloader first.")
        return

    # Extract EpIds from filenames to match with CSV
    available_epids = []
    for f in downloaded_files:
        try:
            # Filename format: SEP28k_EpId.mp3
            epid_str = f.split('_')[1].split('.')[0]
            available_epids.append(int(epid_str))
        except:
            continue

    # Filter CSV for only the audio we actually have
    df_available = df[df['EpId'].isin(available_epids)]
    
    print(f"Episodes found on disk: {len(available_epids)}")
    print(f"Total labeled clips to extract: {len(df_available)}")
    print("Extracting 3-second segments...")

    # Group by EpId so we only open the large MP3 file once per episode
    grouped = df_available.groupby('EpId')
    
    success_count = 0
    fail_count = 0

    for ep_id, clips in tqdm(grouped, desc="Processing Episodes"):
        # Find the specific file for this EpId
        audio_file = next((f for f in downloaded_files if f.startswith(f"SEP28k_{ep_id}.")), None)
        if not audio_file:
            continue
            
        path = os.path.join(AUDIO_DIR, audio_file)
        
        try:
            # Load the audio file (16kHz is industry standard for speech AI)
            # librosa will use the FFmpeg you just installed automatically
            y, sr = librosa.load(path, sr=16000)
            total_samples = len(y)
            
            for _, row in clips.iterrows():
                clip_id = row['ClipId']
                start_sample = int(row['Start'])
                stop_sample = int(row['Stop'])
                
                # Boundary Safety Checks
                if start_sample >= total_samples or start_sample >= stop_sample:
                    continue
                if stop_sample > total_samples:
                    stop_sample = total_samples
                
                # Slice the array
                clip_audio = y[start_sample:stop_sample]
                
                # Only save if the clip actually has audio data (min 0.1 sec)
                if len(clip_audio) < 1600: 
                    continue
                
                # Clean show name for filename safety
                show_clean = "".join(x for x in str(row['Show']) if x.isalnum())
                clip_name = f"SEP28k_{show_clean}_{ep_id}_{clip_id}.wav"
                save_path = os.path.join(CLIPS_OUTPUT, clip_name)
                
                # Save as high-quality WAV for training
                sf.write(save_path, clip_audio, sr)
                success_count += 1
                
        except Exception:
            # If one episode is corrupted, skip it and keep going
            fail_count += 1
            continue

    print("\n" + "="*40)
    print(f"SUCCESS: {success_count} clips generated.")
    print(f"SKIPPED: {fail_count} episodes (corrupted files).")
    print(f"Location: {os.path.abspath(CLIPS_OUTPUT)}")
    print("="*40)

if __name__ == "__main__":
    segment_data()