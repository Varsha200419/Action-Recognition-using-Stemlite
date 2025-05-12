import os
import shutil
import cv2
import numpy as np
from PIL import Image

def augment_frames(video_path, clip_size, frame_rate):
    """
    Sample and augment frames to ensure at least `clip_size` frames. For 2+ frames, use frame averaging (inserting averaged frames
    between originals) and temporal reversal if needed. For fewer than 2 frames, apply a flickering effect to generate clip_size frames.
    Saves to a new folder with '_1' suffix. Frame averaging is the pixel-wise mean of two consecutive frames. Flickering effect adjusts
    brightness to simulate temporal variation.
    Args:
        video_path (str): Path to the folder containing video frames.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames (e.g., every 32nd frame).
    Returns:
        List of frame filenames in the new folder.
    """
    # Create output folder with '_1' suffix
    video_folder_name = os.path.basename(video_path)
    output_dir = os.path.join(os.path.dirname(video_path), f"{video_folder_name}_1")
    os.makedirs(output_dir, exist_ok=True)
    
    all_frames = sorted(os.listdir(video_path))
    sampled_frames = all_frames[::frame_rate]  # Sample every 32nd frame

    if len(sampled_frames) >= clip_size:
        print(f"Skipping augmentation for {video_path}: Already has {len(sampled_frames)} frames.")
        # Copy up to clip_size frames to output folder
        for i, frame in enumerate(sampled_frames[:clip_size]):
            src_path = os.path.join(video_path, frame)
            dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
            shutil.copy(src_path, dst_path)
        return [f"{i:04d}.jpg" for i in range(len(sampled_frames[:clip_size]))]

    print(f"Augmenting frames for {video_path}: Expected {clip_size}, found {len(sampled_frames)}.")
    
    # Load sampled frames as PIL Images
    frames = []
    for frame in sampled_frames:
        src_path = os.path.join(video_path, frame)
        img = Image.open(src_path)
        frames.append(img)
    
    # Handle cases with fewer than 2 frames using flickering effect
    if len(frames) < 2:
        augmented_frames = []
        if len(frames) == 1:
            # Use the single frame
            base_frame = frames[0]
        else:
            # Create a black frame for zero frames
            base_frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))  # Placeholder: 224x224 black image
        
        # Generate clip_size frames with flickering effect
        for i in range(clip_size):
            # Apply brightness variation (e.g., scale pixel values by 0.8 to 1.2)
            brightness_factor = 0.8 + (i % 5) * 0.1  # Cycle through 0.8, 0.9, 1.0, 1.1, 1.2
            frame_np = np.asarray(base_frame).astype(np.float32)
            frame_np = frame_np * brightness_factor
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            augmented_frames.append(Image.fromarray(frame_np))
        
        # Save augmented frames
        for i, frame in enumerate(augmented_frames):
            dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
            frame_np = np.asarray(frame)
            if frame_np.shape[-1] == 4:  # Handle RGBA if present
                frame_np = frame_np[..., :3]
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, frame_bgr)
        
        print(f"Saved {len(augmented_frames)} frames to {output_dir} with flickering effect.")
        return [f"{i:04d}.jpg" for i in range(len(augmented_frames))]
    
    # For 2+ frames, use frame averaging and temporal reversal
    augmented_frames = []
    for i in range(len(frames)):
        # Add the current original frame
        augmented_frames.append(frames[i])
        
        # If not the last frame, add the averaged frame between frames[i] and frames[i+1]
        if i < len(frames) - 1:
            img_1 = np.asarray(frames[i])
            img_2 = np.asarray(frames[i + 1])
            # Compute the average
            img_avg = np.mean([img_1, img_2], axis=0, dtype=np.uint8)
            augmented_frames.append(Image.fromarray(np.uint8(img_avg)))
    
    # If clip_size is not reached, use temporal reversal
    while len(augmented_frames) < clip_size:
        # Create a reversed copy of the current augmented_frames
        reversed_frames = augmented_frames[::-1]
        # Append the reversed frames
        augmented_frames.extend(reversed_frames)
    
    # Trim to clip_size
    augmented_frames = augmented_frames[:clip_size]
    
    # Save augmented frames to output folder
    for i, frame in enumerate(augmented_frames):
        dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
        # Convert PIL Image to BGR for OpenCV
        frame_np = np.asarray(frame)
        if frame_np.shape[-1] == 4:  # Handle RGBA if present
            frame_np = frame_np[..., :3]
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, frame_bgr)
    
    print(f"Saved {len(augmented_frames)} frames to {output_dir} with sequential numbering.")
    return [f"{i:04d}.jpg" for i in range(len(augmented_frames))]

def preprocess_dataset(root_dir, clip_size=8, frame_rate=16):
    """
    Preprocess the dataset by sampling/augmenting frames and saving to new '_1' folders, then delete original folders.
    Args:
        root_dir (str): Path to the dataset root directory.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames.
    """
    subfolders = os.listdir(root_dir)
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        video_folders = sorted(os.listdir(subfolder_path))
        for video_folder in video_folders:
            video_path = os.path.join(subfolder_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            all_frames = sorted(os.listdir(video_path))
            sampled_frames = all_frames[::frame_rate]

            # Process frames and save to new '_1' folder
            try:
                augment_frames(video_path, clip_size, frame_rate)
                
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
            except ValueError as e:
                print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    dataset_path = "/user/HS402/zs00774/Downloads/HMDB_simp"
    preprocess_dataset(dataset_path)
