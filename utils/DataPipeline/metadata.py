from tqdm import tqdm
import pandas as pd
import os

def rename_path(row):
    dataset_name = row["dataset"]
    path = row["og_path"]
    
    if dataset_name in ["mt", "snn"]:
        return os.path.splitext(os.path.basename(path))[0]
    elif dataset_name == "core":
        return path.split(os.sep)[1]
    else:
        return os.path.splitext(path)[0]

def augmented_samples_metadata(df):
    replicated_rows = []
    for _, row in tqdm(df.iterrows(), total = len(df), desc = "Replicating metadata entries for augmentations"):
        if row['dataset'] in ['core', 'sem']:
            parent = row['id'].split(".")[0]
        elif row['dataset'] in ['mt', 'snn']:
            parent = row['scene_id'].split("_")[0]
    
        original_row = row.copy()
        original_row['augmented'] = 0
        original_row['parent'] = parent
        replicated_rows.append(original_row)
        for i in range(1, int(row['aug adj']) + 1):
            replicated_row = row.copy()
            replicated_row['id'] = f"{row['id']}_{i}"
            replicated_row['augmented'] = 1
            replicated_row['parent'] = parent
            replicated_rows.append(replicated_row)

    out = pd.DataFrame(replicated_rows)

    return out

def assign_split(df, samples_per_pair, split_label, target_datasets=None):

    entries = df.copy()

    if 'split' not in entries.columns:
        entries['split'] = None
    else:
        entries.loc[entries['split'] == split_label, 'split'] = None

    if target_datasets is not None:
        current_entries = entries[entries['dataset'].isin(target_datasets) & entries['split'].isnull()].copy()
    else:
        current_entries = entries[entries['split'].isnull()].copy()

    current_entries['pair'] = current_entries['dataset'] + "_" + current_entries['category']
    pairs = current_entries['pair'].unique()

    for pair in tqdm(pairs, total=len(pairs), desc = f"Assigning splits"):
        pair_count = 0
        pair_entries = current_entries[current_entries['pair'] == pair]
        parent_sizes = pair_entries['parent'].value_counts().to_dict()
        while pair_count < samples_per_pair and parent_sizes:
            remaining = samples_per_pair - pair_count
            closest_parent = min(parent_sizes, key=lambda x: abs(parent_sizes[x] - remaining))
            entries.loc[entries['parent'] == closest_parent, 'split'] = split_label
            parent_indices = entries[entries['parent'] == closest_parent].index.tolist()
            pair_count += len(parent_indices)
            del parent_sizes[closest_parent]

    entries = entries.drop_duplicates(subset=["id"], keep="first")

    return entries

def subsample_splits(df, subsample_counts_df):
    subsampled_splits = []
    for split, adj_col in zip(['train', 'val', 'test'], ['train adj', 'val adj', 'test adj']):
        df_subset = df[df['split'] == split]

        # Merge with limits
        split_with_limits = df_subset.merge(
            subsample_counts_df[['dataset', 'category', adj_col]],
            on=['dataset', 'category'],
            how='inner'
        ).rename(columns={adj_col: 'limit'})

        # Initialize an empty list for storing subsampled data
        subsampled_group = []

        # Iterate through unique dataset and class combinations
        for (dataset, clss), group in split_with_limits.groupby(['dataset', 'category']):
            limit = int(group['limit'].iloc[0])  # Get the sampling limit
            sample_count = min(len(group), limit)  # Determine the sample count
            subsampled_group.append(group.sample(n=sample_count, random_state=42))

        # Combine subsampled data for this split
        subsampled_splits.append(pd.concat(subsampled_group, ignore_index=True))

    # Combine all splits into a single DataFrame
    subsampled_df = pd.concat(subsampled_splits, ignore_index=True)
    
    return subsampled_df

def global_subsample(df, count):
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    global_subsampled_df = shuffled_df.head(count)
    return global_subsampled_df

