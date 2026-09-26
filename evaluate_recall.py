import pandas as pd
import os

def evaluate_blocking_recall(
    candidate_file="candidate_pairs.tsv", 
    ground_truth_file=os.path.join("dataset", "train", "train_ground_truth.tsv")
):
    print("Loading files...")
    
    # 1. Load Ground Truth
    gt_df = pd.read_csv(ground_truth_file, sep="\t", dtype=str).fillna("")
    truth_map = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        # Split by comma and remove empty strings
        matches = {m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()}
        truth_map[s1_id] = matches

    # 2. Load Generated Candidates
    cand_df = pd.read_csv(candidate_file, sep="\t", dtype=str).fillna("")
    cand_map = {}
    for _, row in cand_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        candidates = {c.strip() for c in str(row.get("candidate_entity_ids", "")).split(",") if c.strip()}
        cand_map[s1_id] = candidates

    # 3. Calculate Metrics
    total_true_pairs = 0
    total_covered_pairs = 0
    total_missed_pairs = 0
    
    for s1_id, true_matches in truth_map.items():
        if not true_matches:
            continue
            
        total_true_pairs += len(true_matches)
        
        # Get candidates generated for this S1 entity (default to empty set if missing)
        generated_candidates = cand_map.get(s1_id, set())
        
        # Intersection of true matches and generated candidates
        covered = true_matches.intersection(generated_candidates)
        total_covered_pairs += len(covered)
        total_missed_pairs += (len(true_matches) - len(covered))

    # 4. Print Report
    recall = total_covered_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    
    print("\n" + "="*30)
    print("BLOCKING EVALUATION REPORT")
    print("="*30)
    print(f"Total S1 entities in GT    : {len(truth_map)}")
    print(f"Total True Pairs           : {total_true_pairs}")
    print(f"Covered Pairs (Hits)       : {total_covered_pairs}")
    print(f"Missed Pairs               : {total_missed_pairs}")
    print("-" * 30)
    print(f"Recall (Pairs Completeness): {recall:.4%} ({recall:.4f})")
    print("="*30)

if __name__ == "__main__":
    evaluate_blocking_recall()