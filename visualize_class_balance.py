import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Updated filename based on user's processed folder
SYNCED_CSV = r'F:\speech_to_text_predictor\data\processed\synced_standardized_labels.csv'
# Fallback path if the above isn't found
ALT_SYNCED_CSV = r'F:\speech_to_text_predictor\data\processed\synced_final_labels.csv'

# Balanced Folders for "After" stats
BALANCED_DIR = r'F:\speech_to_text_predictor\data\processed\balanced_dataset'

# Where to save the resulting graph
SAVE_PATH = r'F:\speech_to_text_predictor\class_balance_graph.png'

def get_stats():
    """Gathers data from CSV and physical folders."""
    categories = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
    
    # 1. Determine which CSV exists
    csv_to_use = SYNCED_CSV if os.path.exists(SYNCED_CSV) else ALT_SYNCED_CSV
    
    original_counts = []
    if os.path.exists(csv_to_use):
        print(f"Reading original stats from: {os.path.basename(csv_to_use)}")
        df = pd.read_csv(csv_to_use)
        for cat in categories:
            if cat in df.columns:
                original_counts.append(int(df[cat].sum()))
            else:
                original_counts.append(0)
    else:
        print(f"Warning: Metadata CSV not found. Checked: {SYNCED_CSV} and {ALT_SYNCED_CSV}")
        original_counts = [0] * len(categories)

    # 2. Get Balanced Counts from Folders
    balanced_counts = []
    for cat in categories:
        cat_path = os.path.join(BALANCED_DIR, cat)
        if os.path.exists(cat_path):
            count = len([f for f in os.listdir(cat_path) if f.endswith('.wav')])
            balanced_counts.append(count)
        else:
            balanced_counts.append(0)
            
    return categories, original_counts, balanced_counts

def create_plot(categories, original, balanced):
    """Creates a side-by-side bar chart."""
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot bars
    bar1 = ax.bar(x - width/2, original, width, label='Original (Imbalanced)', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bar2 = ax.bar(x + width/2, balanced, width, label='Balanced (Current)', color='#4ECDC4', alpha=0.8, edgecolor='black')

    # Styling
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Balancing Report: Before vs. After', fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add numeric labels on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_labels(bar1)
    add_labels(bar2)

    plt.tight_layout()
    
    # Save the file
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"SUCCESS: Visualization saved to: {SAVE_PATH}")

if __name__ == "__main__":
    print("Scanning dataset for visualization...")
    cats, orig, bal = get_stats()
    create_plot(cats, orig, bal)