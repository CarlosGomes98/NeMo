import argparse
import numpy as np
import torch
from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.models.flux.model import ClipConfig, T5Config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate empty encodings for CLIP and T5 models")
    parser.add_argument(
        "--clip-version",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model version to use"
    )
    parser.add_argument(
        "--t5-version",
        type=str,
        default="google/t5-v1_1-xxl",
        help="T5 model version to use"
    )
    parser.add_argument(
        "--clip-max-length",
        type=int,
        default=77,
        help="Maximum sequence length for CLIP"
    )
    parser.add_argument(
        "--t5-max-length",
        type=int,
        default=512,
        help="Maximum sequence length for T5"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="empty_encodings",
        help="Output directory for empty encodings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda', or specific device like 'cuda:0')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"CLIP version: {args.clip_version}")
    print(f"T5 version: {args.t5_version}")

    clip = FrozenCLIPEmbedder(
        version=args.clip_version,
        max_length=args.clip_max_length,
        always_return_pooled=True,
        device=device,
    )
    t5 = FrozenT5Embedder(
        version=args.t5_version,
        max_length=args.t5_max_length,
        device=device,
    )

    clip_encoding = clip.encode("")
    t5_encoding = t5.encode("").transpose(0, 1)

    np.save(f"{args.output_dir}/empty_pooled_embeds.npy", clip_encoding)
    np.save(f"{args.output_dir}/empty_embeds.npy", t5_encoding)
    
    print(f"CLIP encodings saved to: {args.output_dir}/empty_pooled_embeds.npy")
    print(f"T5 encodings saved to: {args.output_dir}/empty_embeds.npy")
    print(f"CLIP encoding shape: {clip_encoding.shape}")
    print(f"T5 encoding shape: {t5_encoding.shape}")


if __name__ == "__main__":
    main()
