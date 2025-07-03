import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
import xml.etree.ElementTree as ET
import traceback

from smol.core import smol

def extract_samples_core()->pd.DataFrame:
    taxonomy_json_path = os.path.join(smol.get_config("data", "SHAPENET_CORE_PATH"), "taxonomy.json")
    with open(taxonomy_json_path, "r") as file:
        taxonomy_json = json.load(file)
    _df = pd.DataFrame(columns = ["id", "path", "dataset", "category", "scene_id", "vertices"])
    for entry in tqdm(taxonomy_json):
        entry_metadata = entry["metadata"]
        dir_name = entry_metadata["name"]
        category = entry_metadata["label"].split(",")[0]
        try:
            for subdir in os.scandir(os.path.join(smol.get_config("data","SHAPENET_CORE_PATH"), dir_name)):
                if subdir.is_dir():
                    path = os.path.join(smol.get_config("data", "SHAPENET_CORE_PATH"), dir_name, subdir.name, "models", "model_normalized.obj")
                    _id = f"{subdir.name}.core.{category}"
                    if os.path.exists(path):
                        new_data = pd.DataFrame([[_id, path, "core", category, "", ""]], columns=_df.columns)
                        _df = pd.concat([_df, new_data], ignore_index=True)
                    else:
                        smol.logger.error(f"Path {path} does not exist!")
        except Exception as e:
            smol.logger.error(e)
    return _df

def extract_samples_sem()->pd.DataFrame:
    metadata_df_path = os.path.join(smol.get_config("data", "SHAPENET_SEM_PATH"), "metadata.csv")
    metadata_df = pd.read_csv(metadata_df_path, sep = ",")
    _df = pd.DataFrame(columns = ["id","path", "dataset", "category", "scene_id", "vertices"])
    for i, row in tqdm(metadata_df.iterrows()):
        filename_name = row["fullId"].split(".")[1]
        if row["category"] is None or pd.isna(row["category"]):
            smol.logger.warning(f"Object {row['fullId']} category is nan")
            continue 
        category = row["category"].split(",")[0]
        path = os.path.join(smol.get_config("data", "SHAPENET_SEM_PATH"), "models-OBJ", "models", f"{filename_name}.obj")
        if os.path.exists(path):
            _id = f"{filename_name}.sem.{category}"
            new_data = pd.DataFrame([[_id, path, "sem", category, "", ""]], columns=_df.columns)
            _df = pd.concat([_df, new_data], ignore_index=True)
        else:
            smol.logger.error(f"Path {path} does not exist!")
    return _df

def extract_samples_mt()->pd.DataFrame:
    mt_root = smol.get_config("data", "MATTERPORT_PATH")
    _df = pd.DataFrame(columns = ["id","path", "dataset", "category", "scene_id", "vertices"])

    for scene_folder in tqdm(os.listdir(mt_root), desc="Scene progress"):
        scene_path = os.path.join(mt_root, scene_folder, "region_segmentations", scene_folder, "region_segmentations")
        if not os.path.exists(scene_path):
            continue
        for region_file in tqdm(os.listdir(scene_path), desc="File progress"):
            if region_file.endswith(".ply"):
                region_num = region_file.split('region')[-1].split('.')[0]
                ply_file_path = os.path.join(scene_path, region_file)
                semseg_file_path = os.path.join(scene_path, f"region{region_num}.semseg.json")
                fsegs_file_path = os.path.join(scene_path, f"region{region_num}.fsegs.json")

                if not (os.path.exists(semseg_file_path) and os.path.exists(fsegs_file_path)):
                    continue

                try:
                    with open(semseg_file_path, 'r') as f:
                        semseg_data = json.load(f)
                except json.JSONDecodeError as e:
                    smol.logger.error(f"Error decoding JSON in file {semseg_file_path}: {e}")
                    continue

                try:
                    with open(fsegs_file_path, 'r') as f:
                        fsegs_data = json.load(f)
                except json.JSONDecodeError as e:
                    smol.logger.error(f"Error decoding JSON in file {fsegs_file_path}: {e}")
                    continue

                try:
                    ply_data = PlyData.read(ply_file_path)
                except Exception as e:
                    smol.logger.error(f"Error reading ply file {ply_file_path}: {e}")
                    continue

                face_segments = fsegs_data["segIndices"]
                face_data = ply_data['face'].data

                for group in semseg_data["segGroups"]:
                    category = group["label"]
                    segments = group["segments"]
                    object_id = group["objectId"]
                    face_mask = np.isin(face_segments, segments)
                    new_faces = face_data[face_mask]
                    vertex_indices_list = [face['vertex_indices'] for face in new_faces]
                    new_vertex_indices = np.unique(np.concatenate(vertex_indices_list))

                    _id = f"{category}_region{region_num}_{scene_folder}_{object_id}"
                    json_str = json.dumps(new_vertex_indices.tolist())
                    new_data = pd.DataFrame([[_id, ply_file_path, "mt", category, scene_folder, json_str]], columns=_df.columns)
                    _df = pd.concat([_df, new_data], ignore_index=True)

    return _df

def extract_samples_snn()->pd.DataFrame:
    snn_root = smol.get_config("data", "SCENENN_PATH")
    _df = pd.DataFrame(columns = ["id","path", "dataset", "category", "scene_id", "vertices"])

    for scene_folder in tqdm(os.listdir(snn_root), desc = "Scene progress"):
        scene_number = os.path.basename(scene_folder)
        if os.path.isfile(scene_folder):
            continue
        ply_file = os.path.join(snn_root, scene_folder, f'{scene_number}.ply')
        xml_file = os.path.join(snn_root, scene_folder, f'{scene_number}.xml')
        try:
            ply_data = PlyData.read(ply_file)
            vertices = ply_data['vertex'].data
        except Exception as e:
            smol.logger.error(f"Error reading ply file {ply_file}: {e}")
            continue
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            smol.logger.error(f"Error reading xml file {xml_file}: {e}")
            continue

        labels = {}
        for label in tqdm(root.findall('label'), desc = "XML progress"):
            nyu_class = label.get('nyu_class')
            label_id = int(label.get('id'))
            if nyu_class not in labels:
                labels[nyu_class] = []
            labels[nyu_class].append(label_id)

        for class_name, label_ids in tqdm(labels.items(), desc = "File progress"):
            for i, label_id in enumerate(label_ids):
                obj_vertices_indices = np.where(vertices['label'] == label_id)[0]
                if len(obj_vertices_indices) == 0:
                    continue
                
                _id = f"{scene_number}_{class_name}_{i}"
                json_str = json.dumps(obj_vertices_indices.tolist())
                new_data = pd.DataFrame([[_id, ply_file, "snn", class_name, scene_folder, json_str]], columns=_df.columns)
                _df = pd.concat([_df, new_data], ignore_index=True)

    return _df