#!./tensorflow/bin/python

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

def generate_label_report(base_directory):
    base_path = Path(base_directory)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{base_directory}' does not exist.")
        return

    print("=" * 60)
    print(f"LABEL REPORT FOR: {base_path.resolve()}")
    print("=" * 60)

    # Track grand totals across the entire dataset
    grand_totals = Counter()
    has_subdirs = False

    # Iterate through subdirectories (1 level deep)
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        has_subdirs = True
        subdir_counts = Counter()
        xml_count = 0

        # Scan for XML files inside this specific subdirectory
        for file_path in subdir.glob("*.xml"):
            xml_count += 1
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Find all tagged objects in the Pascal VOC file
                for obj in root.iter("object"):
                    name_element = obj.find("name")
                    if name_element is not None and name_element.text:
                        label_name = name_element.text.strip()
                        subdir_counts[label_name] += 1
                        grand_totals[label_name] += 1
            except (ET.ParseError, PermissionError):
                # Skip broken or locked XML files gracefully
                continue

        # Print the breakdown for this subdirectory
        print(f"\n📂 Subdirectory: {subdir.name}/ ({xml_count} XML files found)")
        if subdir_counts:
            # Sort labels alphabetically for neatness
            for label, count in sorted(subdir_counts.items()):
                print(f"  └── 🏷️  {label}: {count}")
        else:
            print("  └── ⚠️  No object labels found in this folder.")

    # Print the master Summary if any subdirectories were processed
    if has_subdirs:
        sumOfAllCounts = 0
        print("\n" + "=" * 60)
        print("📊 GRAND TOTALS ACROSS ALL SUBDIRECTORIES")
        print("=" * 60)
        if grand_totals:
            for label, count in sorted(grand_totals.items()):
                sumOfAllCounts += count
                print(f" TOTAL {label}: {count}")
        else:
            print(" No labels were found anywhere.")

        print(f" GRAND TOTAL: {sumOfAllCounts}")
        print("=" * 60)
    else:
        print("\nNo subdirectories found 1 level deep. Check your directory path!")

if __name__ == "__main__":
    # Replace '.' with the actual path to your root folder if running from somewhere else
    # e.g., target_folder = "/home/user/tree_dataset"
    if len(sys.argv) < 2:
        target_folder = "." 
    elif len(sys.argv) == 2:
        target_folder = sys.argv[1]
    else:
        print("Usage: python count_labels.py <path to directory>")
    generate_label_report(target_folder)


