import os
import pandas as pd
import random
import librosa
import soundfile as sf
import shutil
import numpy as np
from tqdm import tqdm

# ==========================================
# PATH CONFIGURATION
# ==========================================
SYNC_CSV = r'F:\speech_to_text_predictor\data\processed\synced_standardized_labels.csv'
SYNC_DIR = r'F:\speech_to_text_predictor\data\processed\standardized_audio'
BALANCED_DIR = r'F:\speech_to_text_predictor\data\processed\balanced_dataset'

# TARGET: 10,000 samples per class
TARGET = 10000 

def augment_audio(y, sr):
    """
    Creates high-quality synthetic variations of stuttering clips.
    """
    choice = random.choice(['pitch', 'speed', 'noise'])
    
    if choice == 'pitch':
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
    elif choice == 'speed':
        return librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))
    else:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        return y + noise_amp * np.random.normal(size=y.shape)

def main():
    print(f"--- Rebuilding Balanced Dataset (Target: {TARGET} per class) ---")
    
    if not os.path.exists(SYNC_CSV):
        print(f"Error: Synced CSV not found at {SYNC_CSV}.")
        return

    # RE-CREATE DIRECTORY STRUCTURE
    if os.path.exists(BALANCED_DIR):
        print("Cleaning existing balanced directory...")
        shutil.rmtree(BALANCED_DIR)
    
    os.makedirs(BALANCED_DIR)
    print(f"Created fresh directory at: {BALANCED_DIR}")

    df = pd.read_csv(SYNC_CSV)
    stutter_types = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
    
    for s_type in stutter_types:
        class_dir = os.path.join(BALANCED_DIR, s_type)
        os.makedirs(class_dir, exist_ok=True)
        
        # Filter data for this class
        sub_df = df[df[s_type] == 1]
        available_rows = sub_df.to_dict('records')
        current_count = len(available_rows)
        
        if current_count == 0:
            print(f"Skipping {s_type}: No original samples found in CSV.")
            continue

        print(f"\nProcessing {s_type}: {current_count} source samples.")

        # CASE 1: UNDERSAMPLING
        if current_count >= TARGET:
            selected = random.sample(available_rows, TARGET)
            for row in tqdm(selected, desc=f"Undersampling {s_type}"):
                show = "".join(x for x in str(row['Show']) if x.isalnum())
                fname = f"SEP28k_{show}_{row['EpId']}_{row['ClipId']}.wav"
                src = os.path.join(SYNC_DIR, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(class_dir, fname))

        # CASE 2: OVERSAMPLING & AUGMENTATION
        else:
            # 1. Copy all originals
            for row in available_rows:
                show = "".join(x for x in str(row['Show']) if x.isalnum())
                fname = f"SEP28k_{show}_{row['EpId']}_{row['ClipId']}.wav"
                src = os.path.join(SYNC_DIR, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(class_dir, fname))
            
            # 2. Augment to fill the gap
            existing_files = [n for n in os.listdir(class_dir) if n.endswith('.wav')]
            needed = TARGET - len(existing_files)
            
            if needed > 0:
                for i in tqdm(range(needed), desc=f"Augmenting {s_type}"):
                    row = random.choice(available_rows)
                    show = "".join(x for x in str(row['Show']) if x.isalnum())
                    fname = f"SEP28k_{show}_{row['EpId']}_{row['ClipId']}.wav"
                    src = os.path.join(SYNC_DIR, fname)
                    
                    try:
                        y, sr = librosa.load(src, sr=16000)
                        y_aug = augment_audio(y, sr)
                        aug_fname = f"aug_{i}_{fname}"
                        sf.write(os.path.join(class_dir, aug_fname), y_aug, sr)
                    except Exception:
                        continue

    print("\n" + "="*40)
    print(f"REBUILD COMPLETE: All folders recreated in {BALANCED_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()
    