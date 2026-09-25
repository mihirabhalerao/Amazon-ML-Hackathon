import pandas as pd

dir="processed_data_sample/stage0"
for split, d in [("train", dir), ("test", dir)]:
        for src in ["source1", "source2", "source3"]:
            key = f"{split}_{src}"
            path = f"{d}/{split}_{src}.tsv"
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            df.columns = df.columns.str.strip()
            print(f"Loaded {key}: {len(df)} records")
            print(df.head(20).to_string(index=False))