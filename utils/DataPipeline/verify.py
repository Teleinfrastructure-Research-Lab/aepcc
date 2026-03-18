import pandas as pd
import matplotlib.pyplot as plt
import os

def augmented_stats(df):
    class_augmented_counts = (
        df
        .groupby(['category', 'augmented'])
        .size()
        .unstack(fill_value=0)
    )
    ax = class_augmented_counts.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 8),
        colormap='tab10',
        xlabel='category',
        ylabel='Count',
        title='Augmented vs. Original Counts per Class'
    )
    plt.legend(title='Augmented', labels=['Original (0)', 'Augmented (1)'], loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def verify(df):
    duplicates_count = df['id'].duplicated().sum()
    print(f"Number of duplicate entries in 'path' column: {duplicates_count}")

    # Verify that all entries with the same parent are assigned to the same split
    # Group by parent and check if there is more than one unique split per parent
    parent_split_issues = df.groupby('parent')['split'].nunique()
    invalid_parents = parent_split_issues[parent_split_issues > 1]

    # Print any parents with inconsistent splits
    if not invalid_parents.empty:
        print("Parents with entries assigned to multiple splits:")
        print(invalid_parents)
    else:
        print("All entries with the same parent are assigned to the same split.")

def stats(preprocessed_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    # Fixed dataset order and colors so they stay consistent across all plots
    dataset_order = ['core', 'sem', 'snn', 'mt']
    dataset_colors = {
        'core': '#1f77b4',   # dark blue
        'sem':  '#aec7e8',   # light blue
        'snn':  '#e377c2',   # pink
        'mt':   '#d62728',   # red
    }

    # Add any unexpected dataset names so the function still works
    extra_datasets = [d for d in preprocessed_df['dataset'].dropna().unique() if d not in dataset_order]
    all_datasets = dataset_order + sorted(extra_datasets)

    color_list = [dataset_colors.get(d, None) for d in all_datasets]

    # Basic statistics
    total_entries = len(preprocessed_df)
    train_entries = (preprocessed_df['split'] == 'train').sum()
    database_entries = (preprocessed_df['split'] == 'database').sum()
    validation_entries = (preprocessed_df['split'] == 'val').sum()
    test_entries = (preprocessed_df['split'] == 'test').sum()

    # Number of entries in each dataset
    dataset_counts = preprocessed_df['dataset'].value_counts().reindex(all_datasets, fill_value=0)
    dataset_counts = dataset_counts[dataset_counts > 0]

    # Number of entries in each split from each dataset
    split_dataset_counts = (
        preprocessed_df.groupby(['split', 'dataset'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=all_datasets, fill_value=0)
    )

    # Database class distribution
    if database_entries > 0:
        database_class_distribution = (
            preprocessed_df[preprocessed_df['split'] == 'database']['category']
            .value_counts()
        )

    # ----------------------------
    # Plot 1: split distribution
    # ----------------------------
    if database_entries > 0:
        plt.figure(figsize=(10, 6))
        plt.pie(
            [train_entries, database_entries, validation_entries, test_entries],
            labels=['Train', 'Database', 'Validation', 'Test'],
            autopct='%1.1f%%',
            startangle=140
        )
        plt.title('Distribution of Entries')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.pie(
            [train_entries, validation_entries, test_entries],
            labels=['Train', 'Validation', 'Test'],
            autopct='%1.1f%%',
            startangle=140
        )
        plt.title('Distribution of Entries')
        plt.show()

    # ----------------------------
    # Plot 2: dataset distribution
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.pie(
        dataset_counts.values,
        labels=dataset_counts.index,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title('Distribution of Entries by Dataset')
    plt.show()

    # ----------------------------
    # Plot 3: split x dataset
    # ----------------------------
    split_dataset_counts.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
        color=color_list
    )
    plt.title('Entries in Each Split by Dataset')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Plot 4: class distribution by dataset for each split
    # ----------------------------
    split_class_dataset_distribution = (
        preprocessed_df.groupby(['split', 'category', 'dataset'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=all_datasets, fill_value=0)
    )

    for split in preprocessed_df['split'].dropna().unique():
        split_data = split_class_dataset_distribution.loc[split]

        # Keep all dataset columns in fixed order so colors never shift
        split_data = split_data.reindex(columns=all_datasets, fill_value=0)

        ax = split_data.plot(
            kind='bar',
            stacked=True,
            figsize=(12, 8),
            color=color_list
        )
        ax.set_title(f'Class Distribution by Dataset for {split.capitalize()} Split')
        ax.set_xlabel('category')
        ax.set_ylabel('Count')
        ax.legend(title='Dataset', loc='upper right')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Plot 5: database class distribution
    # ----------------------------
    if database_entries > 0:
        database_class_distribution.plot(kind='bar', figsize=(12, 6))
        plt.title('Class Distribution in Database Split')
        plt.xlabel('category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Print statistics
    # ----------------------------
    print(f"Total entries: {total_entries}")
    print(f"Train entries: {train_entries}")
    print(f"Database entries: {database_entries}")
    print(f"Validation entries: {validation_entries}")
    print(f"Test entries: {test_entries}")
    print(f"Entries per dataset:\n{dataset_counts}")
    print(f"Entries in each split by dataset:\n{split_dataset_counts}")
    if database_entries > 0:
        print(f"Class distribution in database split:\n{database_class_distribution}")