import pandas as pd
import os
import time
import urllib.request
import ssl

# ---------------------------------------------------------
# SETUP PATHS
# ---------------------------------------------------------
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'audio')
SEP_EPISODES = os.path.join(BASE_DIR, 'data', 'raw', 'SEP-28k_episodes.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
# This bypasses SSL certificate issues common on institutional servers
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, save_path, min_size_kb=500):
    """Reliable downloader for SEP-28k podcast hosts."""
    # If file exists and is larger than 500KB, it's a valid download
    if os.path.exists(save_path) and os.path.getsize(save_path) > (min_size_kb * 1024):
        return "EXISTS"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'audio/mpeg,audio/basic,audio/*;q=0.9'
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as response:
            with open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        
        # Final validation
        if os.path.exists(save_path) and os.path.getsize(save_path) > (min_size_kb * 1024):
            return "SUCCESS"
        else:
            if os.path.exists(save_path): os.remove(save_path)
            return "FAILED_EMPTY"
    except Exception as e:
        if os.path.exists(save_path): os.remove(save_path)
        return f"ERR_{type(e).__name__}"

def process_sep28k():
    print("\n" + "="*40)
    print("   SEP-28k FINAL DOWNLOADER")
    print("="*40)
    
    if not os.path.exists(SEP_EPISODES):
        print(f"CRITICAL ERROR: {SEP_EPISODES} not found!")
        return

    # Load CSV (SEP-28k format: URL in index 2, EpId in index 4)
    df = pd.read_csv(SEP_EPISODES, header=None)
    
    total_rows = len(df)
    print(f"Total episodes to check from CSV: {total_rows}")
    
    for index, row in df.iterrows():
        try:
            url = str(row[2]).strip().split(' ')[0]
            ep_id = "".join(x for x in str(row[4]) if x.isalnum() or x in "-_").strip()
            
            if not url.startswith('http'): continue
            
            filename = f"SEP28k_{ep_id}.mp3"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            result = download_file(url, save_path)
            
            if result == "SUCCESS":
                print(f"[{index}/{total_rows}] NEW DOWNLOAD: {filename}")
            elif result == "EXISTS":
                if index % 50 == 0: 
                    print(f"[{index}/{total_rows}] Already verified on disk.")
            else:
                if index % 20 == 0:
                    print(f"[{index}/{total_rows}] Skipping (Link likely dead or host down)")
        except Exception:
            continue

    # Final Verification: Count files physically present in the folder
    actual_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("SEP28k")]
    
    print("\n" + "="*40)
    print(f"DOWNLOAD PHASE COMPLETE")
    print(f"Total Rows Processed: {total_rows}")
    print(f"Actual Files on Disk: {len(actual_files)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    process_sep28k()