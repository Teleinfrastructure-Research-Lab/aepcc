import os
import pandas as pd
from smol.core import smol

def apply_synonym(row, synonyms_df):
    """
    Replace the category in a row with its synonym using the synonyms DataFrame.
    If no synonym is found, assign the category as 'REMOVED'.
    """
    category = row["category"]
    synonym_row = synonyms_df[synonyms_df["original"] == category]
    if not synonym_row.empty:
        return synonym_row["synonym"].values[0]
    return "REMOVED"

if __name__ == "__main__":
    # Paths
    core_samples_path = os.path.join(smol.get_config("data", "CORE_SAMPLES_PATH"))
    sem_samples_path = os.path.join(smol.get_config("data", "SEM_SAMPLES_PATH"))
    mt_samples_path = os.path.join(smol.get_config("data", "MT_SAMPLES_PATH"))
    snn_samples_path = os.path.join(smol.get_config("data", "SNN_SAMPLES_PATH"))
    synonyms_path = os.path.join(smol.get_config("data", "SYNONYMS_PATH"))
    output_summary_csv = os.path.join(smol.get_config("paths", "TEST_OUTPUTS_DIR"), "dataset_category_summary.csv")

    # Load DataFrames
    core_df = pd.read_csv(core_samples_path, sep=";")
    sem_df = pd.read_csv(sem_samples_path, sep=";")
    mt_df = pd.read_csv(mt_samples_path, sep=";", on_bad_lines='warn')
    snn_df = pd.read_csv(snn_samples_path, sep=";")
    synonyms_df = pd.read_csv(synonyms_path, sep=";")

    # Combine all with dataset labels
    aggregated_df = pd.concat([core_df, sem_df, mt_df, snn_df], ignore_index=True)

    # Apply synonyms
    aggregated_df["category"] = aggregated_df.apply(
        lambda row: apply_synonym(row, synonyms_df), axis=1
    )

    # Group by dataset and category, then count
    summary_df = aggregated_df.groupby(["dataset", "category"]).size().reset_index(name="count")

    # Sort for readability
    summary_df = summary_df.sort_values(by=["dataset", "count"], ascending=[True, False])

    # Save the summary
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"Dataset-category summary saved at: {output_summary_csv}")
