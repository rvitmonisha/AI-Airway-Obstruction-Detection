import os
import pandas as pd

annotation_folder = "datasets/annotation"

all_labels = []

txt_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]

for txt_file in txt_files:

    annotation_path = os.path.join(annotation_folder, txt_file)

    df = pd.read_csv(
        annotation_path,
        sep="\t",
        header=None,
        names=["Start", "End", "Crackles", "Wheezes"]
    )

    base_name = txt_file.replace(".txt", "")

    for i, row in df.iterrows():

        filename = f"{base_name}_cycle_{i+1}.wav"

        crackles = int(row["Crackles"])
        wheezes = int(row["Wheezes"])

        if crackles == 0 and wheezes == 0:
            label = "Normal"

        elif crackles == 1 and wheezes == 0:
            label = "Crackles"

        elif crackles == 0 and wheezes == 1:
            label = "Wheezes"

        else:
            label = "Both"

        all_labels.append([filename, label])

labels_df = pd.DataFrame(
    all_labels,
    columns=["Filename", "Label"]
)

labels_df.to_csv(
    "labels/labels.csv",
    index=False
)

print("--------------------------------")
print("Labels Created Successfully!")
print(f"Total Labels: {len(labels_df)}")