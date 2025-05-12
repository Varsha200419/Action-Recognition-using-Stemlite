import os
import shutil
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import random

def custom_gaussian_noise(image, sigma=0.1):
    """
    Apply Gaussian noise to a PIL Image.
    Args:
        image (PIL.Image): Input image.
        sigma (float): Standard deviation of the Gaussian noise.
    Returns:
        PIL.Image: Image with added Gaussian noise.
    """
    # Convert PIL Image to tensor
    img_tensor = T.ToTensor()(image)
    # Generate Gaussian noise
    noise = torch.randn_like(img_tensor) * sigma
    # Add noise and clip to [0, 1]
    noisy_tensor = torch.clamp(img_tensor + noise, 0, 1)
    # Convert back to PIL Image
    return T.ToPILImage()(noisy_tensor)

def calculate_optical_flow(prev_frame, next_frame):
    """
    Calculate optical flow between two consecutive frames using OpenCV.
    Args:
        prev_frame (np.ndarray): Previous frame in RGB.
        next_frame (np.ndarray): Next frame in RGB.
    Returns:
        np.ndarray: Optical flow map.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def interpolate_frame(prev_frame, next_frame, flow, alpha=0.5):
    """
    Interpolate a frame between prev_frame and next_frame using optical flow.
    Args:
        prev_frame (np.ndarray): Previous frame in RGB.
        next_frame (np.ndarray): Next frame in RGB.
        flow (np.ndarray): Optical flow map.
        alpha (float): Interpolation factor (0.0 = prev_frame, 1.0 = next_frame).
    Returns:
        np.ndarray: Interpolated frame.
    """
    h, w = prev_frame.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Displace pixels based on flow
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    map_x = x + flow_x * alpha
    map_y = y + flow_y * alpha
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    # Warp the previous frame
    interpolated = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)
    # Blend with next frame
    interpolated = (1 - alpha) * interpolated + alpha * next_frame
    return np.clip(interpolated, 0, 255).astype(np.uint8)

def augment_frames(video_path, clip_size, frame_rate):
    """
    Sample and augment frames to ensure at least `clip_size` frames.
    - Logs frame count after sampling.
    - For <2 frames, applies random subset of PyTorch augmentations (color jitter, crop, Gaussian noise, flip, rotation, scaling).
    - For 2+ frames but <clip_size, uses optical flow-based interpolation.
    Saves to a new folder with '_1' suffix.
    Args:
        video_path (str): Path to the folder containing video frames.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames (e.g., every 32nd frame).
    Returns:
        List of frame filenames in the new folder, number of sampled frames.
    """
    # Create output folder with '_1' suffix
    video_folder_name = os.path.basename(video_path)
    output_dir = os.path.join(os.path.dirname(video_path), f"{video_folder_name}_1")
    os.makedirs(output_dir, exist_ok=True)
    
    all_frames = sorted(os.listdir(video_path))
    sampled_frames = all_frames[::frame_rate]  # Sample every 32nd frame
    num_sampled_frames = len(sampled_frames)
    
    print(f"Folder: {video_path}, Sampled Frames: {num_sampled_frames}")

    # Define PyTorch augmentations
    augmentation_pipeline = [
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        # Custom Gaussian noise transform
        lambda x: custom_gaussian_noise(x, sigma=0.1)
    ]

    if num_sampled_frames >= clip_size:
        print(f"Skipping augmentation for {video_path}: Already has {num_sampled_frames} frames.")
        # Copy up to clip_size frames to output folder
        for i, frame in enumerate(sampled_frames[:clip_size]):
            src_path = os.path.join(video_path, frame)
            dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
            shutil.copy(src_path, dst_path)
        return [f"{i:04d}.jpg" for i in range(len(sampled_frames[:clip_size]))], num_sampled_frames

    print(f"Augmenting frames for {video_path}: Expected {clip_size}, found {num_sampled_frames}.")
    
    # Load sampled frames as PIL Images
    frames = []
    for frame in sampled_frames:
        src_path = os.path.join(video_path, frame)
        try:
            img = Image.open(src_path).convert('RGB')
            frames.append(img)
        except Exception as e:
            print(f"Error loading image {src_path}: {e}")
            continue
    
    # Handle cases with fewer than 2 frames using PyTorch augmentations
    if num_sampled_frames < 2:
        augmented_frames = []
        if num_sampled_frames == 1:
            base_frame = frames[0]
        else:
            # Create a black frame for zero frames
            base_frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Generate clip_size frames with random augmentations
        for i in range(clip_size):
            frame = base_frame
            # Randomly select augmentations to apply
            num_augs = random.randint(1, len(augmentation_pipeline))  # Apply 1 to all augmentations
            selected_augs = random.sample(augmentation_pipeline, num_augs)  # Random subset
            # Apply selected augmentations in random order
            for aug in selected_augs:
                frame = aug(frame)
            augmented_frames.append(frame)
        
        # Save augmented frames
        for i, frame in enumerate(augmented_frames):
            dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
            frame_np = np.asarray(frame)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, frame_bgr)
        
        print(f"Saved {len(augmented_frames)} frames to {output_dir} with random PyTorch augmentations.")
        return [f"{i:04d}.jpg" for i in range(len(augmented_frames))], num_sampled_frames
    
    # For 2+ frames, use optical flow-based interpolation
    augmented_frames = []
    for i in range(len(frames)):
        augmented_frames.append(frames[i])
        if i < len(frames) - 1:
            # Calculate optical flow and interpolate
            prev_np = np.asarray(frames[i])
            next_np = np.asarray(frames[i + 1])
            flow = calculate_optical_flow(prev_np, next_np)
            # Add one interpolated frame at alpha=0.5
            interpolated_np = interpolate_frame(prev_np, next_np, flow, alpha=0.5)
            augmented_frames.append(Image.fromarray(interpolated_np))
    
    # If still short of clip_size, add more interpolated frames
    while len(augmented_frames) < clip_size:
        for i in range(len(frames) - 1):
            prev_np = np.asarray(frames[i])
            next_np = np.asarray(frames[i + 1])
            flow = calculate_optical_flow(prev_np, next_np)
            # Add interpolated frame with random alpha
            alpha = np.random.uniform(0.3, 0.7)
            interpolated_np = interpolate_frame(prev_np, next_np, flow, alpha)
            augmented_frames.append(Image.fromarray(interpolated_np))
            if len(augmented_frames) >= clip_size:
                break
    
    # Trim to clip_size
    augmented_frames = augmented_frames[:clip_size]
    
    # Save augmented frames
    for i, frame in enumerate(augmented_frames):
        dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
        frame_np = np.asarray(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, frame_bgr)
    
    print(f"Saved {len(augmented_frames)} frames to {output_dir} with optical flow interpolation.")
    return [f"{i:04d}.jpg" for i in range(len(augmented_frames))], num_sampled_frames

def preprocess_dataset(root_dir, clip_size=8, frame_rate=32):
    """
    Preprocess the dataset by sampling/augmenting frames and saving to new '_1' folders, then delete original folders.
    Logs the number of sampled frames for each folder.
    Args:
        root_dir (str): Path to the dataset root directory.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames.
    Returns:
        dict: Mapping of class -> video folder -> number of sampled frames.
    """
    frame_counts = {}  # Store frame counts: {class: {video_folder: num_frames}}
    subfolders = os.listdir(root_dir)
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        frame_counts[subfolder] = {}
        video_folders = sorted(os.listdir(subfolder_path))
        for video_folder in video_folders:
            video_path = os.path.join(subfolder_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            try:
                # Process frames and get frame count
                frame_filenames, num_sampled_frames = augment_frames(video_path, clip_size, frame_rate)
                frame_counts[subfolder][video_folder] = num_sampled_frames
                
                # Verify the new folder
                new_folder = os.path.join(subfolder_path, f"{video_folder}_1")
                total_frames = len(os.listdir(new_folder)) if os.path.exists(new_folder) else 0
                print(f"Folder: {new_folder}, Total Frames After Preprocessing: {total_frames}")
                
                # Delete the original folder
                try:
                    shutil.rmtree(video_path)
                    print(f"Deleted original folder: {video_path}")
                except Exception as e:
                    print(f"Error deleting {video_path}: {e}")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
    
    # Log frame counts
    print("\nFrame Counts After Sampling (Before Augmentation):")
    for class_name, videos in frame_counts.items():
        print(f"Class: {class_name}")
        for video, count in videos.items():
            print(f"  Video: {video}, Sampled Frames: {count}")
    
    return frame_counts

if __name__ == "__main__":
    dataset_path = "/user/HS402/zs00774/Downloads/HMDB_simp"
    frame_counts = preprocess_dataset(dataset_path)
