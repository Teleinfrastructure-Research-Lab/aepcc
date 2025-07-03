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

    # Statistics
    total_entries = len(preprocessed_df)
    train_entries = len(preprocessed_df[preprocessed_df['split'] == 'train'])
    database_entries = len(preprocessed_df[preprocessed_df['split'] == 'database'])
    validation_entries = len(preprocessed_df[preprocessed_df['split'] == 'val'])
    test_entries = len(preprocessed_df[preprocessed_df['split'] == 'test'])

    # Number of entries in each dataset
    dataset_counts = preprocessed_df['dataset'].value_counts()

    # Number of entries in each split from each dataset
    split_dataset_counts = preprocessed_df.groupby(['split', 'dataset']).size().unstack(fill_value=0)

    if database_entries>0:
        database_class_distribution = preprocessed_df[preprocessed_df['split'] == 'database']['category'].value_counts()


    # Plotting
    if database_entries>0:
        plt.figure(figsize=(10, 6))
        plt.pie([train_entries, database_entries, validation_entries, test_entries], labels=['Train', 'Database', 'Validation', 'Test'], autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Entries')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.pie([train_entries, validation_entries, test_entries], labels=['Train', 'Validation', 'Test'], autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Entries')
        plt.show()

    plt.figure(figsize=(10, 6))
    plt.pie(dataset_counts, labels=dataset_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Entries by Dataset')
    plt.show()

    split_dataset_counts.plot(kind='bar', stacked=True)
    plt.title('Entries in Each Split by Dataset')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

    split_class_dataset_distribution = preprocessed_df.groupby(['split', 'category', 'dataset']).size().unstack(fill_value=0)

    for split in preprocessed_df['split'].unique():
        # Get data for the current split
        split_data = split_class_dataset_distribution.loc[split]
        
        # Plot stacked bar chart
        ax = split_data.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')
        ax.set_title(f'Class Distribution by Dataset for {split.capitalize()} Split')
        ax.set_xlabel('category')
        ax.set_ylabel('Count')
        ax.legend(title='Dataset', loc='upper right')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    if database_entries>0:
        database_class_distribution.plot(kind='bar')
        plt.title('Class Distribution in Database Split')
        plt.xlabel('category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    # Print statistics
    print(f"Total entries: {total_entries}")
    print(f"Train entries: {train_entries}")
    print(f"Database entries: {database_entries}")
    print(f"Validation entries: {validation_entries}")
    print(f"Test entries: {test_entries}")
    print(f"Entries per dataset:\n{dataset_counts}")
    print(f"Entries in each split by dataset:\n{split_dataset_counts}")
    if database_entries>0:
        print(f"Class distribution in database split:\n{database_class_distribution}")
