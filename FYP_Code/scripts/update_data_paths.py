"""
Update CSV file paths from old machine to new machine structure
"""
import pandas as pd
import os
from pathlib import Path

# Define paths
base_dir = Path(r"C:\Users\FOCS3\Documents\GitHub\fyp-project\FYP_Code")
raw_data_dir = base_dir / "data" / "raw" / "COVID-19_Radiography_Dataset"
processed_dir = base_dir / "data" / "processed"

# Class mapping (old CSV format -> new directory structure)
class_mapping = {
    'COVID-19': 'COVID',
    'Normal': 'Normal',
    'Lung_Opacity': 'Lung_Opacity',
    'Viral_Pneumonia': 'Viral Pneumonia'
}

# Label mapping
label_to_class = {
    0: 'COVID-19',
    1: 'Normal',
    2: 'Lung_Opacity',
    3: 'Viral_Pneumonia'
}

def update_csv_paths(csv_file):
    """Update paths in CSV file to match new machine structure"""
    print(f"\nProcessing {csv_file.name}...")

    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"Original rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Create new image paths
    new_paths = []
    missing_count = 0

    for idx, row in df.iterrows():
        class_name = row['class_name']

        # Map to new directory structure
        if class_name in class_mapping:
            new_class_dir = class_mapping[class_name]
        else:
            new_class_dir = class_name

        # Extract image filename from old path
        old_path = Path(row['image_path'])
        image_filename = old_path.name

        # Construct new path
        new_path = raw_data_dir / new_class_dir / "images" / image_filename

        # Check if file exists
        if new_path.exists():
            new_paths.append(str(new_path))
        else:
            # Try alternative naming (e.g., COVID-19-1.png vs COVID-1.png)
            alt_filename = image_filename.replace('COVID-19-', 'COVID-')
            alt_path = raw_data_dir / new_class_dir / "images" / alt_filename

            if alt_path.exists():
                new_paths.append(str(alt_path))
            else:
                print(f"  WARNING: File not found: {image_filename}")
                missing_count += 1
                new_paths.append(str(new_path))  # Keep path even if missing

    # Update dataframe
    df['image_path'] = new_paths

    # Save updated CSV
    output_file = csv_file
    df.to_csv(output_file, index=False)
    print(f"[OK] Updated {len(df)} rows")
    print(f"  Missing files: {missing_count}")
    print(f"  Saved to: {output_file}")

    return df, missing_count

if __name__ == "__main__":
    print("="*60)
    print("Updating CSV file paths for new machine")
    print("="*60)

    # List of CSV files to update
    csv_files = [
        processed_dir / "train.csv",
        processed_dir / "val.csv",
        processed_dir / "test.csv",
        processed_dir / "all_data.csv"
    ]

    total_missing = 0

    for csv_file in csv_files:
        if csv_file.exists():
            df, missing = update_csv_paths(csv_file)
            total_missing += missing
        else:
            print(f"\nSkipping {csv_file.name} (not found)")

    print("\n" + "="*60)
    print(f"COMPLETE! Total missing files: {total_missing}")
    if total_missing > 0:
        print("WARNING: Some files were not found. May need to regenerate CSVs.")
    else:
        print("[OK] All paths updated successfully!")
    print("="*60)
