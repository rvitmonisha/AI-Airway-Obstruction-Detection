import os
import pandas as pd

annotation_folder = "datasets/annotation"

# Get all annotation files
txt_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

print("Total annotation files:", len(txt_files))

# Read the first annotation file
first_file = txt_files[0]

print("\nReading:", first_file)

annotation_path = os.path.join(annotation_folder, first_file)

# Read the file
df = pd.read_csv(
    annotation_path,
    sep="\t",
    header=None,
    names=["Start", "End", "Crackles", "Wheezes"]
)

print(df)