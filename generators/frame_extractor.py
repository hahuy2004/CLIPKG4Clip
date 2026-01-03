"""
Frame Extractor Module
Extracts and resizes frames from video files uniformly.
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from videos and resize them to target resolution."""
    
    def __init__(self, dataset_root, target_size=(224, 224), fps=1):
        """
        Initialize Frame Extractor.
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'dataset/MSRVTT')
            target_size: Target frame size (width, height)
            fps: Frames per second to extract (None = extract all frames)
        """
        self.dataset_root = Path(dataset_root)
        self.videos_dir = self.dataset_root / 'videos'
        self.frames_dir = self.dataset_root / 'frames'
        self.target_size = target_size
        self.fps = fps
        
        # Create output directory
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames_from_video(self, video_path, output_dir):
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            
        Returns:
            Number of frames extracted
        """
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return 0
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            if self.fps is None:
                frame_interval = 1  # Extract all frames
            else:
                frame_interval = int(video_fps / self.fps) if video_fps > 0 else 1
                frame_interval = max(1, frame_interval)
            
            # Extract frames
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame at specified interval
                if frame_count % frame_interval == 0:
                    # Resize frame
                    resized_frame = cv2.resize(frame, self.target_size, 
                                             interpolation=cv2.INTER_AREA)
                    
                    # Save frame
                    frame_filename = f"frame_{saved_count:04d}.jpg"
                    frame_path = output_dir / frame_filename
                    cv2.imwrite(str(frame_path), resized_frame)
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.debug(f"Extracted {saved_count} frames from {video_path.name}")
            return saved_count
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return 0
    
    def process_dataset(self, skip_existing=True, whitelist_video_ids=None):
        """
        Process all videos in the dataset.
        
        Args:
            skip_existing: Skip videos that already have extracted frames
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.videos_dir.exists():
            logger.error(f"Videos directory not found: {self.videos_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(self.videos_dir.glob(f'*{ext}')))
            video_files.extend(list(self.videos_dir.glob(f'*{ext.upper()}')))
        
        if not video_files:
            logger.warning(f"No video files found in {self.videos_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        stats = {'processed': 0, 'skipped': 0, 'failed': 0, 'filtered': 0}
        
        # Process each video
        for video_path in tqdm(video_files, desc="Extracting frames"):
            # Get video ID (filename without extension)
            video_id = video_path.stem
            
            # Filter by whitelist if provided
            if whitelist_video_ids is not None and video_id not in whitelist_video_ids:
                logger.debug(f"Filtering out {video_id} (not in training split)")
                stats['filtered'] += 1
                continue
            
            # Create output directory for this video
            output_dir = self.frames_dir / video_id
            
            # Skip if already processed
            if skip_existing and output_dir.exists() and len(list(output_dir.glob('*.jpg'))) > 0:
                logger.debug(f"Skipping {video_id} (frames already exist)")
                stats['skipped'] += 1
                continue
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            num_frames = self.extract_frames_from_video(video_path, output_dir)
            
            if num_frames > 0:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Frame extraction complete. Processed: {stats['processed']}, "
                   f"Skipped: {stats['skipped']}, Failed: {stats['failed']}, "
                   f"Filtered: {stats.get('filtered', 0)}")
        
        return stats


def extract_frames(dataset_name, dataset_root='dataset', fps=1, target_size=(224, 224)):
    """
    Convenience function to extract frames from a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MSRVTT')
        dataset_root: Root directory containing datasets
        fps: Frames per second to extract
        target_size: Target frame size (width, height)
        
    Returns:
        Processing statistics
    """
    dataset_path = Path(dataset_root) / dataset_name
    extractor = FrameExtractor(dataset_path, target_size=target_size, fps=fps)
    return extractor.process_dataset()


if __name__ == "__main__":
    # Test the frame extractor
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--dataset_name', type=str, default='MSRVTT',
                       help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Root directory containing datasets')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames per second to extract')
    parser.add_argument('--size', type=int, default=224,
                       help='Target frame size (square)')
    
    args = parser.parse_args()
    
    stats = extract_frames(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        fps=args.fps,
        target_size=(args.size, args.size)
    )
    
    print(f"\nExtraction complete:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
