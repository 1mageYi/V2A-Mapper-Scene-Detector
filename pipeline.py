import os
import argparse
import cv2
import torch
import clip
from PIL import Image
import numpy as np
from keras.models import load_model
from audioldm import clap_to_audio, build_model, save_wave
import pandas as pd
from utils import load_clip_model, aggregate_video_feature, load_mapper

def extract_clip_features(video_path, model, preprocess, device, batch_size=30):
    """Extracts CLIP features from a video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    frame_features = []
    batch_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)
        batch_frames.append(image_input)

        if len(batch_frames) == batch_size:
            batch_frames = torch.cat(batch_frames, dim=0)
            with torch.no_grad():
                image_features = model.encode_image(batch_frames)
            frame_features.append(image_features)
            batch_frames = []

    if batch_frames:
        batch_frames = torch.cat(batch_frames, dim=0)
        with torch.no_grad():
            image_features = model.encode_image(batch_frames)
        frame_features.append(image_features)

    cap.release()
    torch.cuda.synchronize()

    frame_features = torch.cat(frame_features, dim=0).cpu().numpy()
    return frame_features


def predict_clap(video_feature, mapper, device):
    """Predicts CLAP features using the PyTorch MLP mapper."""
    video_feature = torch.from_numpy(video_feature).float().to(device)
    video_feature = video_feature.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        clap = mapper(video_feature).cpu().numpy()
    return clap

def inference_single(clap, audioldm):
    """Performs inference using AudioLDM with the given CLAP features."""
    reshaped_data = clap.reshape(1, 1, -1)
    tensor_data = torch.from_numpy(reshaped_data).float().to('cuda:0')
    with torch.no_grad():
        waveform = clap_to_audio(
            latent_diffusion=audioldm,
            clap=tensor_data,
            text='',
            seed=42,
            duration=10,
            guidance_scale=4.5,
            n_candidate_gen_per_text=3,
        )
    return waveform

def process_folder(folder_path, output_path, model, preprocess, mapper, audioldm, device):
    """Processes all video files in a folder for inference."""
    
    # Take test set out
    with open('/public/home/qinxy/yimj/V2A/file_list.txt', 'r') as f:
        test_files = [line.strip() for line in f.readlines()]


    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            video_id = os.path.splitext(filename)[0]
            if f"{video_id}.wav" in test_files:
                frame_features = extract_clip_features(video_path, model, preprocess, device)
                if frame_features is None:
                    continue

                video_feature = aggregate_video_feature(frame_features)
                predicted_clap = predict_clap(video_feature, mapper,device)
                waveform = inference_single(predicted_clap[0], audioldm)
                save_wave(waveform, output_path, video_id)

def main():
    parser = argparse.ArgumentParser(description="Run inference on video files using CLIP and AudioLDM.")
    parser.add_argument("--mapper_path", type=str, required=True, help="Path to the V2A mapper checkpoint.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing video files.")
    parser.add_argument("--output", type=str, required=True, help="Output folder for generated audio files.")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for processing video frames.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = load_clip_model(device)
    mapper = load_mapper(args.mapper_path,device)
    with torch.no_grad():
        audioldm = build_model(model_name='audioldm-s-full')
    
    process_folder(args.video_folder, args.output, model, preprocess, mapper, audioldm, device)

if __name__ == "__main__":
    main()
