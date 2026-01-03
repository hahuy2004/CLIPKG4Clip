"""
Feature Extractor Module
Extracts CLIP visual features from video frames.
"""
import os
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor:
    """Extract CLIP features from video frames."""
    
    def __init__(self, dataset_root, model_name='ViT-B/32', device='cuda'):
        """
        Initialize CLIP Feature Extractor.
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'dataset/MSRVTT')
            model_name: CLIP model variant to use
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.dataset_root = Path(dataset_root)
        self.frames_dir = self.dataset_root / 'frames'
        self.features_dir = self.dataset_root / 'features'
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Create output directory
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def extract_features_from_frames(self, frames_dir):
        """
        Extract CLIP features from all frames in a directory.
        
        Args:
            frames_dir: Directory containing frame images
            
        Returns:
            numpy array of features with shape (num_frames, feature_dim)
        """
        # Get all frame files
        frame_files = sorted(list(frames_dir.glob('*.jpg')) + 
                           list(frames_dir.glob('*.png')))
        
        if not frame_files:
            logger.warning(f"No frames found in {frames_dir}")
            return None
        
        features_list = []
        
        # Process frames in batches for efficiency
        batch_size = 32
        num_batches = (len(frame_files) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(frame_files))
                batch_files = frame_files[start_idx:end_idx]
                
                # Load and preprocess images
                images = []
                for frame_file in batch_files:
                    try:
                        image = Image.open(frame_file).convert('RGB')
                        image = self.preprocess(image)
                        images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to load frame {frame_file}: {str(e)}")
                        # Use a blank image as placeholder
                        images.append(torch.zeros(3, 224, 224))
                
                if not images:
                    continue
                
                # Stack images into batch
                image_batch = torch.stack(images).to(self.device)
                
                # Extract features
                features = self.model.encode_image(image_batch)
                features = features.cpu().numpy()
                features_list.append(features)
        
        if not features_list:
            return None
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=0)
        
        return all_features
    
    def process_video(self, video_id, skip_existing=True):
        """
        Process a single video to extract features.
        
        Args:
            video_id: Video identifier
            skip_existing: Skip if features already exist
            
        Returns:
            True if successful, False otherwise
        """
        # Check if features already exist
        output_path = self.features_dir / f"{video_id}.npy"
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping {video_id} (features already exist)")
            return True
        
        # Get frames directory
        frames_dir = self.frames_dir / video_id
        if not frames_dir.exists():
            logger.warning(f"Frames directory not found for {video_id}")
            return False
        
        try:
            # Extract features
            features = self.extract_features_from_frames(frames_dir)
            
            if features is None:
                logger.error(f"Failed to extract features for {video_id}")
                return False
            
            # Save features
            np.save(output_path, features)
            logger.debug(f"Saved features for {video_id}: shape {features.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {video_id}: {str(e)}")
            return False
    
    def process_dataset(self, skip_existing=True, whitelist_video_ids=None):
        """
        Process all videos in the dataset.
        
        Args:
            skip_existing: Skip videos that already have features
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.frames_dir.exists():
            logger.error(f"Frames directory not found: {self.frames_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        # Get all video frame directories
        video_dirs = [d for d in self.frames_dir.iterdir() if d.is_dir()]
        
        if not video_dirs:
            logger.warning(f"No video frame directories found in {self.frames_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        logger.info(f"Found {len(video_dirs)} videos to process")
        
        stats = {'processed': 0, 'skipped': 0, 'failed': 0, 'filtered': 0}
        
        # Process each video
        for video_dir in tqdm(video_dirs, desc="Extracting CLIP features"):
            video_id = video_dir.name
            
            # Filter by whitelist if provided
            if whitelist_video_ids is not None and video_id not in whitelist_video_ids:
                logger.debug(f"Filtering out {video_id} (not in training split)")
                stats['filtered'] += 1
                continue
            
            # Check if already processed
            if skip_existing and (self.features_dir / f"{video_id}.npy").exists():
                stats['skipped'] += 1
                continue
            
            # Process video
            success = self.process_video(video_id, skip_existing=False)
            
            if success:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Feature extraction complete. Processed: {stats['processed']}, "
                   f"Skipped: {stats['skipped']}, Failed: {stats['failed']}, "
                   f"Filtered: {stats.get('filtered', 0)}")
        
        return stats


def extract_features(dataset_name, dataset_root='datasets', device='cuda'):
    """
    Convenience function to extract CLIP features from a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MSRVTT')
        dataset_root: Root directory containing datasets
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Processing statistics
    """
    dataset_path = Path(dataset_root) / dataset_name
    extractor = CLIPFeatureExtractor(dataset_path, device=device)
    return extractor.process_dataset()


if __name__ == "__main__":
    # Test the feature extractor
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CLIP features from frames')
    parser.add_argument('--dataset_name', type=str, default='MSRVTT',
                       help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='datasets',
                       help='Root directory containing datasets')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    stats = extract_features(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        device=args.device
    )
    
    print(f"\nFeature extraction complete:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
