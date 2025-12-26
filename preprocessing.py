import os
import pandas as pd
import soundfile as sf
import random

# ---------------- CONFIG ----------------
CSV_PATH = "data/fluencybank_labels.csv"
AUDIO_DIR = "data/audio"
OUTPUT_DIR = "processed"
SAMPLE_RATE = 16000
RANDOM_SEED = 42
# ---------------------------------------

random.seed(RANDOM_SEED)

# Label vote columns
LABEL_COLUMNS = [
    "Unsure",
    "PoorAudioQuality",
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "DifficultToUnderstand",
    "Interjection",
    "NoStutteredWords",
    "NaturalPause",
    "Music",
    "NoSpeech"
]

# 1️⃣ Load CSV
df = pd.read_csv(CSV_PATH)

# 2️⃣ Assign class by max vote (SEGMENT LABELING)
def get_class(row):
    votes = row[LABEL_COLUMNS]
    if votes.sum() == 0:
        return None
    return votes.idxmax()

df["Class"] = df.apply(get_class, axis=1)
df = df.dropna(subset=["Class"])

print("✅ Segments labeled")

# 3️⃣ CLASS BALANCING (UNDERSAMPLING)
class_counts = df["Class"].value_counts()
min_count = class_counts.min()

print("\n📊 Original class distribution:")
print(class_counts)

balanced_rows = []

for cls in class_counts.index:
    cls_rows = df[df["Class"] == cls]
    sampled = cls_rows.sample(n=min_count, random_state=RANDOM_SEED)
    balanced_rows.append(sampled)

balanced_df = pd.concat(balanced_rows).sample(frac=1, random_state=RANDOM_SEED)

print("\n⚖️ Balanced class distribution:")
print(balanced_df["Class"].value_counts())

# 4️⃣ Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in balanced_df["Class"].unique():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# 5️⃣ AUDIO PREPROCESSING (CROPPING & SAVING)
for _, row in balanced_df.iterrows():

    ep_id = row["EpId"]
    clip_id = row["ClipId"]
    start = int(row["Start"])
    stop = int(row["Stop"])
    cls = row["Class"]

    audio_path = os.path.join(AUDIO_DIR, f"{ep_id}.wav")
    if not os.path.exists(audio_path):
        continue

    audio, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        continue

    clip_audio = audio[start:stop]

    out_name = f"ep{ep_id}_clip{clip_id}.wav"
    out_path = os.path.join(OUTPUT_DIR, cls, out_name)

    sf.write(out_path, clip_audio, SAMPLE_RATE)

print("\n🎯 Segmentation, balancing, and preprocessing COMPLETE")
