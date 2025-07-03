import sys
from pathlib import Path
from smol.core import smol
import traceback

from utils.DataPipeline.extract import extract_samples_core, extract_samples_sem, extract_samples_mt, extract_samples_snn


if __name__ == "__main__":

    dataset = ""
    if len(sys.argv) < 2:
        dataset = "core"
        smol.logger.warning("Dataset argument not provided to script. Running with Core as default!")
    else:
        if sys.argv[1] in ["core", "sem", "mt", "snn"]:
            dataset = sys.argv[1]
        else:
            smol.logger.error(f"Invalid value for dataset argument: {sys.argv[1]}")
            smol.logger.debug(traceback.format_exc())
            quit()

    output_path = ""
    if len(sys.argv) < 3:
        output_path = smol.get_config("data",f"{dataset.upper()}_SAMPLES_PATH")
        try:
            output_path = Path(output_path)
            output_path.resolve(strict=False)
        except Exception as e:
            smol.logger.error(f"Path {output_path} is not valid")
            smol.logger.debug(traceback.format_exc())
            quit()
    else:
        output_path = sys.argv[2]
        try:
            output_path = Path(output_path)
            output_path.resolve(strict=False)
        except Exception as e:
            smol.logger.error(f"Path {output_path} is not valid")
            smol.logger.debug(traceback.format_exc())
            quit()

    if dataset == "core":
        df = extract_samples_core()
    elif dataset == "sem":
        df = extract_samples_sem()
    elif dataset == "mt":
        df = extract_samples_mt()
    elif dataset == "snn":
        df = extract_samples_snn()
    else:
        smol.logger.error(f"Invalid value for dataset argument: {dataset}")
        smol.logger.debug(traceback.format_exc())
        quit()
    
    df.to_csv(output_path, sep = ";", index = False)