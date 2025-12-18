import csv

IN_CSV  = r"data\raw\SEP28k_clean_labels.csv"
OUT_CSV = r"data\raw\SEP28k_binary_labels.csv"

# Columns that indicate stuttering in SEP-28K
STUTTER_COLS = [
    "SoundRep",
    "WordRep",
    "Prolongation",
    "Block",
    "Interjection"
]

def is_one(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")

total = 0
stutter = 0
fluent = 0

with open(IN_CSV, "r", newline="", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames + ["binary_label"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            label = 0
            for c in STUTTER_COLS:
                if c in row and is_one(row[c]):
                    label = 1
                    break

            row["binary_label"] = label
            writer.writerow(row)

            if label == 1:
                stutter += 1
            else:
                fluent += 1

print("✅ Binary labels created")
print("Total rows:", total)
print("Stutter (1):", stutter)
print("Fluent (0):", fluent)
print("Saved to:", OUT_CSV)
