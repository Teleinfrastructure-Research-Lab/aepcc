#!/usr/bin/env python3
# scene_codec.py
#
# Codec for multi-object point-cloud scenes.
# -------------------------------------------------------------------
# Encode:
#   scene_codec.py -m encode  scene.obj  scene.bin
#
# Decode:
#   scene_codec.py -m decode  scene.bin  recon.obj
#
# Both (encode ➜ decode ➜ RD metrics):
#   scene_codec.py --mode both -a tgae -d 256 -q 8 scene.obj recon.obj
# -------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import click
import numpy as np
import torch
import trimesh
import pathlib
import time


from smol.core import smol
from utils.codec_sa import get_encoder, get_decoder
from utils.codec import (
    quantize8_tensor, quantize_and_pack_4bit, quantize_and_pack_2bit,
    quantize_and_pack_1bit, dequantize8_tensor, unpack_and_dequantize_4bit,
    unpack_and_dequantize_2bit, unpack_and_dequantize_1bit,
    compress_scene, decompress_scene, remove_isolated_points
)
from utils.geometry import sphere_normalization_masked_with_params, sphere_denormalization
from utils.utils import d1_psnr_pcc
from utils.visualize import visualize_point_cloud_o3d

REMOVE_IDX_FROM_IMR = True

# ================================================================
# -------- time helper -------------------------------------
# ================================================================
def _now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

# ================================================================
# -------- scene I/O helpers -------------------------------------
# ================================================================
def load_vertex_groups(path: str | pathlib.Path,
                       num_pts: int = 2048
                      ) -> list[tuple[np.ndarray, torch.Tensor]]:
    """
    Return one (points[N,3], mask[1,N]) tuple per `g`/`o` section of a
    *vertex-only* OBJ file.

    Works even when the file has **no faces**.
    """
    objects, cur = [], []

    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if line.startswith(('g ', 'o ')):          # ---- new group ----
                if cur:
                    objects.append(np.asarray(cur, np.float32))
                    cur = []
            elif line.startswith('v '):                # vertex line
                x, y, z = map(float, line.split()[1:4])
                cur.append((x, y, z))

    if cur:                                           # last group
        objects.append(np.asarray(cur, np.float32))

    out = []
    for P in objects:
        M = len(P)
        if M > num_pts:                               # subsample
            idx    = np.random.choice(M, num_pts, False)
            P_sub  = P[idx]
            mask   = torch.ones(1, num_pts)
        else:                                         # pad
            pad    = np.zeros((num_pts - M, 3), np.float32)
            P_sub  = np.vstack([P, pad])
            mask   = torch.cat([torch.ones(1, M),
                                torch.zeros(1, num_pts - M)], dim=1)
        out.append((P_sub, mask))

    return out

def save_pointcloud_scene(objects: List[np.ndarray], path: str):

    with open(path, 'w') as f:
        f.write("# point-cloud scene exported by scene_codec\n")
        for i, P in enumerate(objects, 1):
            f.write(f"g object_{i}\n")
            for x, y, z in P:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

def save_pointcloud_objects(objects : Sequence[np.ndarray], out_dir : str | pathlib.Path, fmt : str = "ply") -> List[pathlib.Path]:

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, P in enumerate(objects, 1):
        P = np.asarray(P, np.float32)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"object {i} has wrong shape {P.shape}")

        path = out_dir / f"object_{i}.{fmt.lower()}"

        if fmt.lower() == "ply":
            trimesh.PointCloud(P).export(path, encoding="binary_little_endian")
        elif fmt.lower() == "obj":
            with open(path, "w") as f:
                f.write("# point cloud exported by save_pointcloud_objects\n")
                f.write(f"o object_{i}\n")
                for x, y, z in P:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                f.write("p 1\n")
        else:
            raise ValueError("fmt must be 'ply' or 'obj'")
        written.append(path.resolve())
    return written

# ================================================================
# -------- IMR handling helpers ----------------------------------
# ================================================================
def _encode_object(pc: torch.Tensor, mask: torch.Tensor, encoder,
                   architecture: str, qbits: int, qstats: np.ndarray,
                   device) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    pc, mask = pc.to(device), mask.to(device)
    with torch.no_grad():
        imr = encoder(pc, mask) if architecture == 'foldingnet' \
              else encoder(pc, None, mask)

    def _q(t):
        return (t if qbits == 32 else
                t.half() if qbits == 16 else
                quantize8_tensor(t, qstats) if qbits == 8 else
                quantize_and_pack_4bit(t, qstats) if qbits == 4 else
                quantize_and_pack_2bit(t, qstats) if qbits == 2 else
                quantize_and_pack_1bit(t, qstats))

    if architecture in ('foldingnet', 'tgae'):
        return _q(imr)
    return (_q(imr[0]), imr[1].short())          # GAE

def _dequantise(imr_q, architecture, qbits, qstats, shape):
    def _dq(t):
        return (t if qbits == 32 else
                t.float() if qbits == 16 else
                dequantize8_tensor(t, qstats) if qbits == 8 else
                unpack_and_dequantize_4bit(t, qstats, shape) if qbits == 4 else
                unpack_and_dequantize_2bit(t, qstats, shape) if qbits == 2 else
                unpack_and_dequantize_1bit(t, qstats, shape))
    return _dq(imr_q) if architecture in ('foldingnet', 'tgae') \
           else (_dq(imr_q[0]), imr_q[1].squeeze(0).long())

def _pack_imr(imr_q, centroid, max_dist):
    d = {"centroid": centroid, "max_distance": max_dist}
    if isinstance(imr_q, (list, tuple)):
        for i, t in enumerate(imr_q, 1):
            d[str(i)] = t
    else:
        d["1"] = imr_q
    return d

def _unpack_imr(d):
    centroid, max_dist = d["centroid"], d["max_distance"]
    imrs = tuple(v for k, v in sorted(d.items()) if k.isdigit())
    return (imrs[0] if len(imrs) == 1 else imrs), centroid, max_dist

def drop_key(items: list[dict], key: str) -> list[dict]:
    return [{k: v for k, v in d.items() if k != key} for d in items]

# ================================================================
# ---------------- CLI -------------------------------------------
# ================================================================
def _to_int(_, __, value: str) -> int:
    return int(value)
@click.command()
@click.option('-a', '--architecture', type=click.Choice(['foldingnet', 'gae', 'tgae']),
              default='foldingnet', show_default=True)
@click.option("-d", "--vector-size", type=click.Choice(["128", "256", "512"]), default="128", callback=_to_int, show_default=True, help="latent dimension")
@click.option('-q', '--quantization', type=click.Choice(['32','16','8','4','2','1']),
              default='32', show_default=True)
@click.option('-n', '--num-points', default=2048, show_default=True,
              help='points per object sent to the codec')
@click.option('-m', '--mode', default='both',
              type=click.Choice(['encode','decode','both']), show_default=True)
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_file', type=click.Path(dir_okay=False))
def main(architecture, vector_size, quantization,
         num_points, mode, input_file, output_file):

    quantization = int(quantization)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- load models & quant-stats --------------------------------
    encoder = get_encoder(architecture, vector_size).eval().to(device)
    decoder = get_decoder(architecture, vector_size).eval().to(device)
    qstats = np.load(os.path.join(
        smol.get_config("data","quantization_stats"),
        f"{architecture}_{vector_size}.npy"))

    pc_in_scene = pc_out_scene = mask_scene = None
    objs_meta: List[Dict[str, torch.Tensor]] = []

    # ================================================= encode ======
    if mode in ('encode','both'):
        t_enc0 = _now()

        objects = load_vertex_groups(input_file, num_points)
        smol.logger.info(f"Found {len(objects)} objects.")

        for P_np, mask in objects:
            pts_norm, centroid, max_d = sphere_normalization_masked_with_params(torch.from_numpy(P_np), mask.squeeze(0))
            pts = pts_norm.T.unsqueeze(0)

            imr_q = _encode_object(
                pts, mask, encoder, architecture,
                quantization, qstats, device)

            objs_meta.append(_pack_imr(imr_q, centroid, max_d))
            P_torch = torch.Tensor(P_np)
            pc_in_scene = P_torch.T.unsqueeze(0) if pc_in_scene is None else \
                torch.cat((pc_in_scene, P_torch.T.unsqueeze(0)), dim=2)
            mask_scene = mask if mask_scene is None else \
                torch.cat((mask_scene, mask), dim=1)

        bitstream = compress_scene(objs_meta)
        if mode == 'encode':
            Path(output_file).write_bytes(bitstream)

        t_enc1 = _now()
        enc_time_s = t_enc1 - t_enc0
        smol.logger.info(f"Encode time : {enc_time_s:.3f} s")
        try:
            n_pts_in = int(mask_scene.sum().item()) if mask_scene is not None else 0
            if n_pts_in > 0:
                smol.logger.info(f"Encode throughput : {n_pts_in/enc_time_s:,.0f} pts/s")
        except Exception:
            pass

    # ================================================= decode ======
    if mode in ('decode','both'):
        t_dec0 = _now()

        if mode == 'decode':
            bitstream = Path(input_file).read_bytes()

        objs_meta = decompress_scene(bitstream)
        smol.logger.info(f"Bit-stream contains {len(objs_meta)} objects.")

        recon_objects = []
        for obj in objs_meta:
            imr_q, centroid, max_d = _unpack_imr(obj)
            imr = _dequantise(imr_q, architecture, quantization,
                            qstats, (1, vector_size))

            with torch.no_grad():
                if architecture in ('foldingnet','tgae'):
                    pc = decoder(imr.to(device))
                else:  # GAE
                    pc = decoder(imr[0].to(device), imr[1].to(device))
                    pc = remove_isolated_points(pc, imr[1])

            centroid, max_d = centroid.to(pc.device), max_d.to(pc.device)

            pts = pc.squeeze(0)
            if pts.shape[1] != 3:
                pts = pts.T
            pts_world = sphere_denormalization(pts, centroid, max_d).cpu().numpy()
            recon_objects.append(pts_world)

            pc_out_scene = torch.from_numpy(pts_world.T).unsqueeze(0) if pc_out_scene is None else \
                        torch.cat((pc_out_scene,
                                    torch.from_numpy(pts_world.T).unsqueeze(0)), dim=2)

        save_pointcloud_scene(recon_objects, output_file)

        t_dec1 = _now()
        dec_time_s = t_dec1 - t_dec0
        smol.logger.info(f"Decode time : {dec_time_s:.3f} s")
        try:
            n_pts_out = int(pc_out_scene.shape[-1]) if pc_out_scene is not None else 0
            if n_pts_out > 0:
                smol.logger.info(f"Decode throughput : {n_pts_out/dec_time_s:,.0f} pts/s")
        except Exception:
            pass


    # ================================================= RD metrics ==
    if mode == 'both':
        if architecture == "gae" and REMOVE_IDX_FROM_IMR == True:
            bits = len(compress_scene(drop_key(decompress_scene(bitstream), "2")))*8
        else:
            bits = len(bitstream) * 8
        # bpp  = bits / mask_scene.sum()
        bpp  = bits / pc_out_scene.shape[-1]
        smol.logger.info(f"Scene rate : {bpp:.4f} bit/point")

        psnr = d1_psnr_pcc(pc_in_scene[:,:,mask_scene[0,:]==1.0], pc_out_scene, chunk=2048, device='cuda' if torch.cuda.is_available() else 'cpu')
        smol.logger.info(f"Scene PSNR : {psnr:.2f} dB")

        sparsification = 100.0 * (1.0 - (n_pts_out / n_pts_in))
        sparsification = max(0.0, min(100.0, sparsification))
        smol.logger.info(f"Sparsification : {sparsification:.2f} %")

        # visualize_point_cloud_o3d(pc_in_scene.squeeze(0).cpu())
        # visualize_point_cloud_o3d(pc_out_scene.squeeze(0).cpu())
        # import open3d as o3d; o3d.io.write_point_cloud(f"inputs//vpcc_draco_inputs//{os.path.basename(input_file).split('.')[0]}.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_in_scene.squeeze(0).cpu().T.numpy().astype('float32'))), write_ascii=True)


    if mode == 'decode':
        visualize_point_cloud_o3d(pc_out_scene.squeeze(0).cpu())


# ------------------------------------------------------------------
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
