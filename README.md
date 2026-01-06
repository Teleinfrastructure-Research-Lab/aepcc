# Requirements

- Python: 3.10
- CUDA: 12.1 for GPU-enabled PyTorch builds
- Datasets (download and provide paths in `config/out_of_project_paths.yaml`):
  - ShapeNet-Core
  - ShapeNet-Semantic
  - Matterport3D
  - SceneNN
- smol: use commit `214a773` (see the `smol` submodule or install from the provided subfolder). Reference: https://github.com/Teleinfrastructure-Research-Lab/smol/tree/214a773b7a953972cc39d0c7c9edc31f6add2808

# Install

```bash
git clone --recurse-submodules https://github.com/Teleinfrastructure-Research-Lab/aepcc.git
python install_dependencies.py
pip install -e .
pip install submodules/smol
mv config/out_of_project_paths.yaml.example config/out_of_project_paths.yaml
mv config/checkpoints.yaml.example config/checkpoints.yaml
# enter the correct paths
mkdir runs
mkdir runs/foldingnet
mkdir runs/gae
mkdir runs/tgae
mkdir data
```
# Release

The release includes:
- Pretrained checkpoints for all three architectures at latent sizes 128, 256, and 512 (best validation loss).
- Quantization scales used during encoding/decoding.
- `synonyms_250325_reduced.csv` for aggregating categories across datasets (required when generating datasets).

After downloading, extract the archive into the `data/` directory.

# Codebase

- `architecture/`: Model implementations. Includes building blocks and network definitions.
- `codec_sa/`: Standalone codec scripts for single objects and multi-object scenes.
- `dataloaders/`: Data loading and preprocessing utilities (see `DataLoading.py`). Handles reading datasets, masks, normalization, and batching.
- `loss/`: Loss functions and related utilities used during training/evaluation.
- `scripts/`: Helper scripts for maintenance, experimentation, or batch operations outside the main pipeline.
- `tests/`: Minimal tests and examples to validate components.
- `utils/`: Shared helpers for codec and pipeline: quantization/packing, geometry normalization/denormalization, visualization, metrics (PSNR), and common utilities.

# Configuration

Populate `out_of_project_paths.yaml` with the correct paths for the raw dataset samples for ShapeNet-Core, ShapeNet-Semantic, Matterport3D and SceneNN. Provide paths for the `.h5` files, that are going to be generated - **SYNTH** and **FULL**.

# Experiment definitions

The `experiment_definitions/` folder contains ready-to-run YAML configs for training different architectures and latent sizes. Use these with `pipeline/training/train_from_config.py` by passing the directory path, and the trainer (via `smol`) will pick up the YAMLs.

How to modify for training runs:
- Batch size: edit `experiment_definitions/<arch>/batch_config/batch_size_conf.yaml` to fit GPU memory.
- Latent size (F): choose the YAML matching `128/256/512`, or adjust `vector_size` in the chosen YAML.
- Optimizer, LR, epochs, augmentations, dataset split, checkpoint/output paths: open the corresponding YAML and tweak fields as needed; defaults write results under `runs/<arch>/`.
- Model architecture-specific params: adjust model blocks in the YAML (e.g., layers, hidden sizes) consistent with the files in `architecture/`.


# Pipeline

## Data
To generate the **SYNTH** and **FULL** datasets use the scripts under `pipeline/data/`.

1. `extract_samples.py` takes as argument a dataset name `[core, sem, mt, snn]`, uses the paths to the raw datasets provided in the configuration, and outputs a CSV with the paths to samples from this dataset and other metadata. 

2. `generate_metadata.py` takes one argument - `true` for **SYNTH** and `false` for **FULL**. It uses the samples extracted in previous step to form metadata CSV for the dataset to be generated. It performs duplciation, balancing and spliting.

3. `generate_h5.py` takes one argument - `true` for **SYNTH** and `false` for **FULL**. It uses the metadata CSV to fetch samples from the dataset, augment them and add them to a `.h5` file that can be used for training.

## Training
`train_from_config.py` takes a path to experiment definitions directory. Such experiment definitions can be found in the `experiment_definitions` directory. They can be used to train FoldingNet, GAE and TGAE. The script uses `smol` to train the networks and saves the results (checkpoints, loss curves and metadata) to a path defined the experiment definition `.yaml` file. As these paths are defined by default, the results go into the `runs/` directory.

# Rate-Distortion Experiments

The `rate-distortion-compute` directory contains scripts that do rate-distortion experiments for objects and scenes. These are the results reported in Fig. 9 in the manuscript.

Both `rdc_per_object.py` and `rdc_per_scene.py` take as argument a path to experiment configuration directory. These are available in `rate-distortion-compute/rdc_configs`. The scripts save the resutls (rate-distortion samples for different architectures and configuratons) as a CSV in `outputs/rdc_outputs`.

# Standalone Codec

Use `codec_sa/object_codec.py` (single object) and `codec_sa/scene_codec.py` (multi-object scene) to encode, decode, or run both steps with RD reporting.

- `--mode`: one of `[encode, decode, both]` (default `both`).
  - `encode`: input is 3D geometry; output is a compressed binary stream.
  - `decode`: input is a compressed binary stream; output is decoded 3D geometry.
  - `both`: input is 3D geometry; output is decoded geometry (bitstream not saved).

- `--architecture`: `['foldingnet', 'gae', 'tgae']` (default `foldingnet`).
- `--vector-size`: latent dimension F, `[128, 256, 512]` (default `128`).
- `--quantization`: bits per latent element, `['32','16','8','4','2','1']` (default `32`).

Models are loaded from `data/models/` via the project config (quantization stats are also resolved from `data/` as configured).

## Object codec: expected inputs and behavior

Input formats:
- OBJ/PLY mesh with faces: the mesh surface is uniformly sampled to exactly `2048` points (faces are used for sampling; raw vertex lists are not used directly).
- OBJ/PLY point cloud: points are read from the file. Binary little-endian PLY is supported for output; input OBJ/PLY parsing follows the loader’s expected structure.

Point count handling:
- If the point cloud has more than 2048 points, it is subsampled internally to 2048 to fit the model input size.
- If it has fewer than 2048 points, it is padded to 2048; a mask marks real vs. padded points, and padded points do not contribute to rate or metrics.


## Scene codec: expected inputs and behavior

Input format:
- OBJ file containing multiple vertex-only groups. The scene OBJ must be divided into objects via `g <name>` or `o <name>` sections. Faces are not required and are ignored; only `v` lines are used.

Per-object handling:
- Each group/object is extracted as a point cloud. If it has more than 2048 points, it is subsampled to 2048; if fewer, it is padded to 2048 and masked.
- Objects are encoded independently, packed into a single scene bitstream, and decoded back to per-object point clouds.

# Acknowledgements

This research was funded by the European Union–Next Generation EU through the National Recovery and Resilience Plan of the Republic of Bulgaria, Project No. BG-RRP-2.004-0005, “Improving the research capacity and quality to achieve international recognition and resilience of TU-Sofia” (IDEAS). The work was carried out in academic collaboration with Princeton University, facilitated by the EU Horizon 2020 Marie Skłodowska–Curie Research and Innovation Staff Exchange Programme “Research Collaboration and Mobility for Beyond 5G Future Wireless Networks (RECOMBINE)” under Grant Agreement No. 872857. The authors also acknowledge the support of the Teleinfrastructure R&D Laboratory at the Technical University of Sofia and the Intelligent Communication Infrastructures R&D Laboratory at Sofia Tech Park, Sofia, Bulgaria.

# Citation

```
@ARTICLE{bozhilov2025autoencoder,
  author={Bozhilov, Ivaylo and Petkova, Radostina and Tonchev, Krasimir and Manolova, Agata and Poulkov, Vladimir and Vincent Poor, H.},
  journal={IEEE Access}, 
  title={Autoencoder Architectures for Low-Rate Sparse Point Cloud Geometry Coding}, 
  year={2025},
  volume={13},
  number={},
  pages={214122-214140},
  keywords={Point cloud compression;Autoencoders;Geometry;Encoding;Three-dimensional displays;Codecs;Transformers;Computer architecture;Image reconstruction;Decoding;Autoencoder;coding;compression;machine learning;point cloud;source coding},
  doi={10.1109/ACCESS.2025.3646031}}

```
