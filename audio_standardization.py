import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Source of extracted 3-second clips from segment_audio.py
RAW_CLIPS_DIR = r'F:\speech_to_text_predictor\data\processed\clips'
# The cleaned labels file you uploaded
CLEAN_CSV = r'F:\speech_to_text_predictor\data\raw\SEP28k_clean_labels.csv' 
# Destination for standardized audio
OUTPUT_AUDIO_DIR = r'F:\speech_to_text_predictor\data\processed\standardized_audio'
# Destination for the Master Synced CSV
SYNCED_CSV_PATH = r'F:\speech_to_text_predictor\data\processed\synced_standardized_labels.csv'

def main():
    print("Starting Audio Standardization & Master Sync...")
    
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # 1. Load the cleaned labels
    try:
        df = pd.read_csv(CLEAN_CSV)
        print(f"Successfully loaded {len(df)} entries from clean labels.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Identify clips physically present on the F: drive
    available_clips = set(os.listdir(RAW_CLIPS_DIR))
    print(f"Found {len(available_clips)} physical .wav files in clips folder.")

    standardized_rows = []
    
    # 3. Processing Loop
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Standardizing Audio"):
        # Match the filename convention used in segment_audio.py
        show_name = "".join(x for x in str(row['Show']) if x.isalnum())
        fname = f"SEP28k_{show_name}_{row['EpId']}_{row['ClipId']}.wav"
        
        if fname in available_clips:
            in_path = os.path.join(RAW_CLIPS_DIR, fname)
            out_path = os.path.join(OUTPUT_AUDIO_DIR, fname)
            
            try:
                # Load Audio at 16kHz (AI standard)
                y, sr = librosa.load(in_path, sr=16000)
                
                if len(y) > 0:
                    # Peak Normalization: Scale volume so max peak is 1.0
                    # This removes volume bias between different podcast episodes
                    y_norm = librosa.util.normalize(y)
                    
                    # Save the new standardized file
                    sf.write(out_path, y_norm, sr)
                    
                    # Add this row to our "Verified" list
                    standardized_rows.append(row)
            except Exception:
                # If a specific file is corrupted, we skip it
                continue

    # 4. Generate the Final Synced CSV
    if standardized_rows:
        synced_df = pd.DataFrame(standardized_rows)
        synced_df.to_csv(SYNCED_CSV_PATH, index=False)
        
        print("\n" + "="*40)
        print("SUCCESS: Standardization & Sync Complete")
        print(f"Total Clips Synced: {len(synced_df)}")
        print(f"Master CSV Created: {SYNCED_CSV_PATH}")
        print(f"Standardized Audio: {OUTPUT_AUDIO_DIR}")
        print("="*40)
        print("NEXT STEP: Proceed to 'us4_balance_data.py' using the new Synced CSV.")
    else:
        print("\nERROR: No clips were matched or processed. Check your file paths.")

if __name__ == "__main__":
    main()