import click
import os
import glob
import yaml
from torch.utils.tensorboard import SummaryWriter
import traceback
from typing import Optional

from smol.core import smol
from smol.signal import Signal
from smol.tasks.Trainer import BasicRuntimeConfigTrainer

writer:Optional[SummaryWriter] = None

def training_callback(trainer: BasicRuntimeConfigTrainer, signal: Signal, step: int):
    global writer
    running_metrics = trainer.aux_config.running_metrics
    scalar_dict = {}
    for key in running_metrics:
        try:
            scalar = signal.item(key)
            scalar_dict[key] = scalar
        except Exception as e:
            smol.logger.error(f"Skipping {key}: {e}")
    if scalar_dict:
        writer.add_scalars("RunningMetrics", scalar_dict, step)
    

def epoch_callback(trainer: BasicRuntimeConfigTrainer, epoch:int, train_loss:float, val_loss: float, current_lr: float):
    global writer
    writer.add_scalars("Per Epoch Loss", {"Training": train_loss, "Validation": val_loss}, epoch)
    writer.add_scalar("Learning Rate", current_lr, epoch)

@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output_path', type=click.Path(exists=False), default = None, help="Optional output path. Value taken from experiment yaml if not provided.")
def process_config_files(input_path, output_path):
    """Load YAML configs from INPUT_PATH and train using BasicRuntimeConfigTrainer."""
    global writer
    # Find all YAML config files in the input directory
    yaml_files = glob.glob(os.path.join(input_path, "*.yaml"))
    if not yaml_files:
        smol.logger.info("No YAML configuration files found in input path.")
        return
    
    smol.logger.info(f"Found {len(yaml_files)} YAML configuration files.")
    
    # Load all YAML configurations
    smol.register()
    smol.add_runtime_configs(yaml_files)
    
    for yaml_file in yaml_files:
        exp_name = os.path.splitext(os.path.basename(yaml_file))[0]
        try:
            if output_path == None:
                _output_path = smol.get_config(exp_name, "output_path") 
                if _output_path == None or _output_path == "":
                    raise ValueError(f"Output path from {exp_name}.yaml is empty.")
            else:
                _output_path = output_path
            trainer = BasicRuntimeConfigTrainer(exp_name, _output_path)
            writer = SummaryWriter(log_dir=trainer.logs_dir)
            trainer.training_callback = training_callback
            trainer.epoch_callback = epoch_callback
            smol.logger.info(f"Running trainer for {exp_name}...")
            trainer.run()
        except Exception as e:
            smol.logger.error(f"Running experiment {exp_name} failed: {e}")
            smol.logger.debug(traceback.format_exc())
        writer.close()
        writer = None

    smol.logger.info("Training process completed.")

if __name__ == "__main__":
    process_config_files()