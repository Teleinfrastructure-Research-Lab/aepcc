import pandas as pd
import sys
from smol.core import smol
from utils.DataPipeline.verify import stats, augmented_stats, verify


if __name__ == "__main__":
    if len(sys.argv) <2:
        synth_flag = False
    else:
        synth_flag = sys.argv[1].lower() == "true"
    if synth_flag:
        preprocessed_df = pd.read_csv(smol.get_config("data", "METADATA_SYNTH_CSV_PATH"))
    else:
        preprocessed_df = pd.read_csv(smol.get_config("data", "METADATA_FULL_CSV_PATH"))

    stats(preprocessed_df)
    augmented_stats(preprocessed_df)
    verify(preprocessed_df)