import pandas as pd
import os
import sys

from smol.core import smol
from utils.DataPipeline.metadata import augmented_samples_metadata, subsample_splits, global_subsample, assign_split
from utils.DataPipeline.verify import stats,  verify, augmented_stats



if __name__ == "__main__":

    if len(sys.argv) <2:
        synth_flag = False
    else:
        synth_flag = sys.argv[1].lower() == "true"

    core_samples_path = os.path.join(smol.get_config("data", "CORE_SAMPLES_PATH"))
    sem_samples_path = os.path.join(smol.get_config("data", "SEM_SAMPLES_PATH"))
    mt_samples_path = os.path.join(smol.get_config("data", "MT_SAMPLES_PATH"))
    snn_samples_path = os.path.join(smol.get_config("data", "SNN_SAMPLES_PATH"))
    synonyms_path = os.path.join(smol.get_config("data", "SYNONYMS_PATH"))

    mt_snn_val_samples = smol.get_config("data", "MT_SNN_VAL_SAMPLES")
    mt_snn_test_samples = smol.get_config("data", "MT_SNN_TEST_SAMPLES")
    shapenet_val_samples = smol.get_config("data", "SHAPENET_VAL_SAMPLES")
    shapenet_test_asmples = smol.get_config("data", "SHAPENET_TEST_SAMPLES")
    global_samples_count = smol.get_config("data", "GLOBAL_SAMPLES_COUNT")
    global_samples_synth_count = smol.get_config("data", "GLOBAL_SAMPLES_SYNTH_COUNT")

    #STEP 1 READ AND PREPARE DATA -----------------------------------------------------------------------------------------------
    core = pd.read_csv(core_samples_path, sep = ";")
    sem = pd.read_csv(sem_samples_path, sep = ";")
    mt = pd.read_csv(mt_samples_path, sep = ";", on_bad_lines='warn')
    snn = pd.read_csv(snn_samples_path, sep = ";")
    synonyms = pd.read_csv(synonyms_path, sep = ";")

    if synth_flag == False:
        merged_df = pd.concat([core, sem, mt, snn], ignore_index=True) # merge all datasets metadata
    else:
        merged_df = pd.concat([core, sem], ignore_index=True) # merge all datasets metadata
    merged_df["scene_id"] = merged_df["scene_id"].astype(str)


    #STEP 2 APPLY SYNONYNMS -----------------------------------------------------------------------------------------------
    merged_df = merged_df[merged_df['category'].isin(synonyms['original'])] 
    synonym_mapping = dict(zip(synonyms['original'], synonyms['synonym']))
    merged_df['category'] = merged_df['category'].replace(synonym_mapping)

    # STEP 3 load augmentations csv and replicate metadata entries for augmentations---------------------------------------
    if synth_flag == False:
        aug_df = pd.read_csv(smol.get_config("data", "AUG_FULL_CSV_PATH"), sep = ";")
    else:
        aug_df = pd.read_csv(smol.get_config("data", "AUG_SYNTH_CSV_PATH"), sep = ";")
    merged_aug = pd.merge(merged_df, aug_df, on=['dataset', 'category'], how='left')
    augmented_df = augmented_samples_metadata(merged_aug)
    augmented_df.drop(columns=['aug adj'], inplace=True)
    augmented_df = augmented_df.reset_index(drop=True)

    # STEP 4 assign splits ---------------------------------------------------------------------------------------------------
    split_df = augmented_df
    if synth_flag == False:
        split_df = assign_split(split_df, smol.get_config("data", "MT_SNN_VAL_SAMPLES"), "val", ["mt", "snn"]) # val
        split_df = assign_split(split_df, smol.get_config("data", "MT_SNN_TEST_SAMPLES"), "test", ["mt", "snn"]) # test
    else:
        split_df = assign_split(split_df, smol.get_config("data", "SHAPENET_VAL_SAMPLES"), "val", ["core", "sem"]) # val
        split_df = assign_split(split_df, smol.get_config("data", "SHAPENET_TEST_SAMPLES"), "test", ["core", "sem"]) # test
    split_df.loc[split_df['split'].isnull(), 'split'] = 'train' # train

    # STEP 5 load subsample csv and subsample splits -------------------------------------------------------------------------
    if synth_flag == False:
        subsample_num = pd.read_csv(smol.get_config("data", "SUBSAMPLE_FULL_CSV_PATH"), sep = ";")
    else:
        subsample_num = pd.read_csv(smol.get_config("data", "SUBSAMPLE_SYNTH_CSV_PATH"), sep = ";")
    final_subsampled_df = subsample_splits(split_df, subsample_num)
    final_subsampled_df.drop(columns=['limit'], inplace=True)

    # STEP 6 global subsample -------------------------------------------------------------------------------------------------
    if synth_flag == False:
        global_subsampled_df = global_subsample(final_subsampled_df, smol.get_config("data", "GLOBAL_SAMPLES_COUNT"))
    else:
        global_subsampled_df = global_subsample(final_subsampled_df, smol.get_config("data", "GLOBAL_SAMPLES_SYNTH_COUNT"))


    # STEP 7 verify -----------------------------------------------------------------------------------------------------------
    stats(global_subsampled_df)
    augmented_stats(global_subsampled_df)
    verify(global_subsampled_df)

    # STEP 8 Shuffle -----------------------------------------------------------------------------------------------------------
    shuffled_df = global_subsampled_df.sample(frac=1).reset_index(drop=True)

    # STEP 9 save csv -----------------------------------------------------------------------------------------------------------
    if synth_flag == False:
        shuffled_df.to_csv(smol.get_config("data", "METADATA_FULL_CSV_PATH"), index=False)
    else:
        shuffled_df.to_csv(smol.get_config("data", "METADATA_SYNTH_CSV_PATH"), index=False)