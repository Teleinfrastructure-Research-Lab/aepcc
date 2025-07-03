import click
from pathlib import Path
import os
import torch
import numpy as np

from smol.core import smol
from utils.codec_sa import get_decoder, get_encoder, load_and_prepare_pointcloud, save_tensor_as_pcd_ply
from utils.codec import (quantize8_tensor, quantize_and_pack_4bit, quantize_and_pack_2bit, 
                         quantize_and_pack_1bit, dequantize8_tensor, unpack_and_dequantize_4bit, 
                         unpack_and_dequantize_2bit, unpack_and_dequantize_1bit, compress_object, 
                         decompress_object, remove_isolated_points, remove_key)
from utils.visualize import visualize_point_cloud_o3d
from utils.utils import d1_psnr_pcc

REMOVE_IDX_FROM_IMR = True

@click.command()
@click.option('-a', '--architecture', required=False, type=str, default = 'foldingnet', help='Model architecture (e.g., foldingnet, gae, tgae)')
@click.option('-d', '--vector-size', required=False, type=int, default = 128, help='Size of the vector representations (128, 256, 512)')
@click.option('-q', '--quantization', required=False, type=int, default = 32, help='Number of bits to represent vector (1, 2, 4, 8, 16, 32)')
@click.option('-m', '--mode', required=False, type=str, default = 'both', help='encode, decode or both')
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('output_file', type=click.Path(dir_okay=False, path_type=Path))
def encode_obj(architecture, vector_size, quantization, mode, input_file, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = get_encoder(architecture, vector_size).eval().to(device)
    decoder = get_decoder(architecture, vector_size).eval().to(device)
    qstats_fname = f"{architecture}_{vector_size}.npy"
    path = os.path.join(smol.get_config("data", "quantization_stats"), qstats_fname)
    quant_stats = np.load(path)
    
    if mode == "encode" or mode == "both":
        pc_in, mask = load_and_prepare_pointcloud(input_file)
        pc_in = pc_in.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            if architecture == 'foldingnet':
                imr = encoder(pc_in, mask)
            else:
                imr = encoder(pc_in, None, mask)
        if architecture == 'foldingnet' or architecture == 'tgae':
            if quantization == 32:
                pass
            elif quantization == 16:
                imr = imr.half()
            elif quantization == 8:
                imr = quantize8_tensor(imr, quant_stats)
            elif quantization == 4:
                imr = quantize_and_pack_4bit(imr, quant_stats)
            elif quantization == 2:
                imr = quantize_and_pack_2bit(imr, quant_stats)
            elif quantization == 1:
                imr = quantize_and_pack_1bit(imr, quant_stats)
            else:
                raise ValueError("Only 32, 16, 8, 4, 1 bit quantization is supported!")
        else:
            if quantization == 32:
                pass
            elif quantization == 16:
                imr = (imr[0].half(), imr[1])
            elif quantization == 8:
                imr = (quantize8_tensor(imr[0], quant_stats), imr[1])
            elif quantization == 4:
                imr = (quantize_and_pack_4bit(imr[0], quant_stats), imr[1])
            elif quantization == 2:
                imr = (quantize_and_pack_2bit(imr[0], quant_stats), imr[1])
            elif quantization == 1:
                imr = (quantize_and_pack_1bit(imr[0], quant_stats), imr[1])
            else:
                raise ValueError("Only 32, 16, 8, 4, 2, 1 bit quantization is supported!")

            imr = (imr[0], imr[1].short())
        
        imr_dict = {str(i): v for i, v in enumerate(imr if isinstance(imr, (list, tuple)) else [imr], start=1)}
        bitstream = compress_object(imr_dict)
        if mode == "encode":
            with open(output_file, "wb") as f:
                f.write(bitstream)

    if mode == "decode" or mode == "both":
        if mode == "decode":
            with open(input_file, "rb") as f:
                bitstream = f.read()
        decomp_imr_dict = decompress_object(bitstream)
        decomp_imr = tuple(decomp_imr_dict[str(i)] for i in sorted(map(int, decomp_imr_dict))) if len(decomp_imr_dict) > 1 else next(iter(decomp_imr_dict.values()))
        orig_shape = (1, vector_size)

        if architecture == 'foldingnet' or architecture == 'tgae':
            if quantization == 32:
                pass
            elif quantization == 16:
                decomp_imr = decomp_imr.float()
            elif quantization == 8:
                decomp_imr = dequantize8_tensor(decomp_imr, quant_stats)
            elif quantization == 4:
                decomp_imr = unpack_and_dequantize_4bit(decomp_imr, quant_stats, orig_shape)
            elif quantization == 2:
                decomp_imr = unpack_and_dequantize_2bit(decomp_imr, quant_stats, orig_shape)
            elif quantization == 1:
                decomp_imr = unpack_and_dequantize_1bit(decomp_imr, quant_stats, orig_shape)
            else:
                raise ValueError("Only 16, 8, 4, 1 bit quantization is supported!")
            with torch.no_grad():
                pc_out = decoder(decomp_imr.to(device))
        else:
            if quantization == 32:
                pass
            elif quantization == 16:
                decomp_imr = (decomp_imr[0].float(), decomp_imr[1])
            elif quantization == 8:
                decomp_imr = (dequantize8_tensor(decomp_imr[0], quant_stats), decomp_imr[1])
            elif quantization == 4:
                decomp_imr = (unpack_and_dequantize_4bit(decomp_imr[0], quant_stats, orig_shape), decomp_imr[1])
            elif quantization == 2:
                decomp_imr = (unpack_and_dequantize_2bit(decomp_imr[0], quant_stats, orig_shape), decomp_imr[1])
            elif quantization == 1:
                decomp_imr = (unpack_and_dequantize_1bit(decomp_imr[0], quant_stats, orig_shape), decomp_imr[1])
            else:
                raise ValueError("Only 32, 16, 8, 4, 2, 1 bit quantization is supported!")

            decomp_imr = (decomp_imr[0], decomp_imr[1].squeeze(0).long())
            with torch.no_grad():
                pc_out = decoder(decomp_imr[0].to(device), decomp_imr[1].to(device))
                pc_out = remove_isolated_points(pc_out, decomp_imr[1])

        save_tensor_as_pcd_ply(pc_out, output_file)

    if mode == "both" or mode == "encode":
        if architecture == "gae" and REMOVE_IDX_FROM_IMR == True:
            bits = len(compress_object(remove_key(decompress_object(bitstream), "2")))*8
        else:
            bits = len(bitstream) * 8
        points = mask.sum()
        bpp = bits / points
        smol.logger.info(f"Rate: {bpp}")

    if mode == "both":
        psnr = d1_psnr_pcc(pc_in[:,:,mask[0,:]==1.0], pc_out, chunk=2048, device='cuda' if torch.cuda.is_available() else 'cpu')
        smol.logger.info(f"PSNR: {psnr}")
        visualize_point_cloud_o3d(pc_in.squeeze(0).cpu())
        visualize_point_cloud_o3d(pc_out.squeeze(0).cpu())

    if mode == "decode":
        visualize_point_cloud_o3d(pc_out.squeeze(0).cpu())

if __name__ == '__main__':
    encode_obj()
