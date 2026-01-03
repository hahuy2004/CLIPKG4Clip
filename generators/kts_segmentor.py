"""
KTS Segmentor Module
Performs Kernel Temporal Segmentation on video features to detect event boundaries.
"""
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def compute_change_points_kts(features, num_segments=None, threshold=0.5):
    """
    Compute change points using Kernel Temporal Segmentation (simplified version).
    
    Args:
        features: Feature matrix of shape (num_frames, feature_dim)
        num_segments: Target number of segments (if None, use threshold)
        threshold: Threshold for change point detection
        
    Returns:
        List of change point indices
    """
    n = features.shape[0]
    
    if n <= 2:
        return [0, n - 1]
    
    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute segment scores using dynamic programming
    if num_segments is None:
        # Use ruptures library for automatic change point detection
        try:
            import ruptures as rpt
            
            # Use PELT algorithm for change point detection
            algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(features)
            change_points = algo.predict(pen=threshold * 10)
            
            # Remove the last point (which is always n)
            change_points = [0] + [cp - 1 for cp in change_points[:-1]] + [n - 1]
            
        except ImportError:
            logger.warning("ruptures library not available, using simple method")
            # Fallback: simple method based on feature similarity
            change_points = compute_simple_change_points(features, threshold)
    else:
        # Use fixed number of segments
        segment_length = n / num_segments
        change_points = [int(i * segment_length) for i in range(num_segments + 1)]
        change_points[-1] = n - 1
    
    return sorted(list(set(change_points)))


def compute_simple_change_points(features, threshold=0.5):
    """
    Simple change point detection based on cosine similarity.
    
    Args:
        features: Feature matrix of shape (num_frames, feature_dim)
        threshold: Threshold for detecting changes
        
    Returns:
        List of change point indices
    """
    n = features.shape[0]
    
    if n <= 2:
        return [0, n - 1]
    
    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute frame-to-frame similarity
    similarities = np.array([
        np.dot(features[i], features[i + 1])
        for i in range(n - 1)
    ])
    
    # Find local minima (low similarity = potential change points)
    change_points = [0]
    
    # Use adaptive thresholding
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    adaptive_threshold = mean_sim - threshold * std_sim
    
    for i in range(1, len(similarities) - 1):
        # Check if local minimum and below threshold
        if (similarities[i] < similarities[i - 1] and 
            similarities[i] < similarities[i + 1] and
            similarities[i] < adaptive_threshold):
            change_points.append(i + 1)
    
    change_points.append(n - 1)
    
    # Merge segments that are too short
    min_segment_length = max(3, n // 20)  # At least 3 frames or 5% of video
    merged_points = [change_points[0]]
    
    for i in range(1, len(change_points)):
        if change_points[i] - merged_points[-1] >= min_segment_length:
            merged_points.append(change_points[i])
    
    # Ensure last point is included
    if merged_points[-1] != change_points[-1]:
        merged_points.append(change_points[-1])
    
    return merged_points


class KTSSegmentor:
    """Segment videos into events using Kernel Temporal Segmentation."""
    
    def __init__(self, dataset_root, num_segments=None, threshold=0.5):
        """
        Initialize KTS Segmentor.
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'dataset/MSRVTT')
            num_segments: Target number of segments (None for automatic)
            threshold: Threshold for change point detection
        """
        self.dataset_root = Path(dataset_root)
        self.features_dir = self.dataset_root / 'features'
        self.segments_dir = self.dataset_root / 'segments'
        self.num_segments = num_segments
        self.threshold = threshold
        
        # Create output directory
        self.segments_dir.mkdir(parents=True, exist_ok=True)
    
    def segment_video(self, video_id, skip_existing=True):
        """
        Segment a single video into events.
        
        Args:
            video_id: Video identifier
            skip_existing: Skip if segments already exist
            
        Returns:
            List of segment tuples (start_frame, end_frame)
        """
        # Check if segments already exist
        output_path = self.segments_dir / f"{video_id}.json"
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping {video_id} (segments already exist)")
            with open(output_path, 'r') as f:
                data = json.load(f)
                return data['segments']
        
        # Load features
        features_path = self.features_dir / f"{video_id}.npy"
        if not features_path.exists():
            logger.warning(f"Features not found for {video_id}")
            return None
        
        try:
            features = np.load(features_path)
            
            # Detect change points
            change_points = compute_change_points_kts(
                features, 
                num_segments=self.num_segments,
                threshold=self.threshold
            )
            
            # Convert change points to segments
            segments = []
            for i in range(len(change_points) - 1):
                segments.append({
                    'start_frame': int(change_points[i]),
                    'end_frame': int(change_points[i + 1])
                })
            
            # Save segments
            output_data = {
                'video_id': video_id,
                'num_frames': int(features.shape[0]),
                'num_segments': len(segments),
                'segments': segments
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.debug(f"Segmented {video_id} into {len(segments)} events")
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting {video_id}: {str(e)}")
            return None
    
    def process_dataset(self, skip_existing=True, whitelist_video_ids=None):
        """
        Process all videos in the dataset.
        
        Args:
            skip_existing: Skip videos that already have segments
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.features_dir.exists():
            logger.error(f"Features directory not found: {self.features_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        # Get all feature files
        feature_files = list(self.features_dir.glob('*.npy'))
        
        if not feature_files:
            logger.warning(f"No feature files found in {self.features_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        logger.info(f"Found {len(feature_files)} videos to segment")
        
        stats = {'processed': 0, 'skipped': 0, 'failed': 0, 'filtered': 0}
        
        # Process each video
        for feature_file in tqdm(feature_files, desc="Segmenting videos"):
            video_id = feature_file.stem
            
            # Filter by whitelist if provided
            if whitelist_video_ids is not None and video_id not in whitelist_video_ids:
                logger.debug(f"Filtering out {video_id} (not in training split)")
                stats['filtered'] += 1
                continue
            
            # Check if already processed
            if skip_existing and (self.segments_dir / f"{video_id}.json").exists():
                stats['skipped'] += 1
                continue
            
            # Segment video
            segments = self.segment_video(video_id, skip_existing=False)
            
            if segments is not None:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Segmentation complete. Processed: {stats['processed']}, "
                   f"Skipped: {stats['skipped']}, Failed: {stats['failed']}, "
                   f"Filtered: {stats.get('filtered', 0)}")
        
        return stats


def segment_videos(dataset_name, dataset_root='dataset', num_segments=None, threshold=0.5):
    """
    Convenience function to segment videos in a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MSRVTT')
        dataset_root: Root directory containing datasets
        num_segments: Target number of segments per video (None for automatic)
        threshold: Threshold for change point detection
        
    Returns:
        Processing statistics
    """
    dataset_path = Path(dataset_root) / dataset_name
    segmentor = KTSSegmentor(dataset_path, num_segments=num_segments, threshold=threshold)
    return segmentor.process_dataset()


if __name__ == "__main__":
    # Test the segmentor
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment videos using KTS')
    parser.add_argument('--dataset_name', type=str, default='MSRVTT',
                       help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Root directory containing datasets')
    parser.add_argument('--num_segments', type=int, default=None,
                       help='Target number of segments (None for automatic)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for change point detection')
    
    args = parser.parse_args()
    
    stats = segment_videos(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        num_segments=args.num_segments,
        threshold=args.threshold
    )
    
    print(f"\nSegmentation complete:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
