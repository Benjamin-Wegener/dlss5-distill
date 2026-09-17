#!/usr/bin/env python3
"""
Single-image or folder 2x super-resolution inference script.
"""
import os, sys, argparse, time
import torch
from PIL import Image
import numpy as np
from model import Phase4Generator

def parse_args():
    parser = argparse.ArgumentParser(description="2x Real-Time Super-Resolution Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input 540p image")
    parser.add_argument("--output", "-o", type=str, default="output_1080p.png", help="Path to save 1080p image")
    parser.add_argument("--weights", "-w", type=str, default="weights/model_dlss5_distill_deploy.pt", help="Model weights path")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Inference device: mps, cuda, cpu, auto")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device == "auto":
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    print(f"🚀 Running on device: {dev}")

    model = Phase4Generator(deploy=True).to(dev)
    if os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd)
        print(f"✅ Loaded weights from {args.weights}")
    else:
        print(f"⚠️ Weights file {args.weights} not found, running with initial weights")

    model.eval()

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    print(f"Input size: {w}x{h}")

    inp = (torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(dev)

    with torch.no_grad():
        t0 = time.time()
        sr = model(inp)
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000

    out_np = (sr[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_np)
    out_img.save(args.output)
    w_out, h_out = out_img.size
    print(f"✅ Super-resolved to {w_out}x{h_out} in {dt_ms:.2f} ms ({1000.0/dt_ms:.1f} FPS) -> Saved to {args.output}")

if __name__ == "__main__":
    main()
