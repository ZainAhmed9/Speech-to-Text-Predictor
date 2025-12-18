import csv

IN_CSV  = r"data\raw\SEP-28k_labels.csv"
OUT_CSV = r"data\raw\SEP28k_clean_labels.csv"

def is_one(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")

with open(IN_CSV, "r", newline="", encoding="utf-8") as f_in:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames

    if not fieldnames:
        raise SystemExit("CSV empty or header missing.")

    required = ["Unsure", "PoorAudioQuality", "Music"]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    total = 0
    removed = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1

            # Remove if any of these flags == 1
            if is_one(row.get("Unsure")) or is_one(row.get("PoorAudioQuality")) or is_one(row.get("Music")):
                removed += 1
                continue

            # Optional: remove NoSpeech if column exists
            if "NoSpeech" in row and is_one(row.get("NoSpeech")):
                removed += 1
                continue

            writer.writerow(row)

print("✅ Done")
print("Rows before:", total)
print("Rows removed:", removed)
print("Rows after :", total - removed)
print("Saved to   :", OUT_CSV)
