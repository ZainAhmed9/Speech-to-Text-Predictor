import pandas as pd

# ✅ Input / Output paths
IN_CSV  = "data/raw/SEP-28k_labels.csv"
OUT_CSV = "data/raw/SEP28k_clean_labels.csv"

df = pd.read_csv(IN_CSV)
print("✅ Loaded:", IN_CSV)
print("Rows (before):", len(df))

# ✅ 1) Label Cleaning Rules (as per your screenshot)
# Remove rows where Unsure==1 OR PoorAudioQuality==1 OR Music==1
bad_mask = (
    (df["Unsure"] == 1) |
    (df["PoorAudioQuality"] == 1) |
    (df["Music"] == 1)
)

# ✅ (Optional but recommended) Remove clips with NoSpeech==1
# (Noise only / silence only clips)
if "NoSpeech" in df.columns:
    bad_mask = bad_mask | (df["NoSpeech"] == 1)

clean_df = df[~bad_mask].copy()

print("Rows removed:", bad_mask.sum())
print("Rows (after):", len(clean_df))

# ✅ Save cleaned CSV
clean_df.to_csv(OUT_CSV, index=False)
print("✅ Saved clean labels to:", OUT_CSV)
