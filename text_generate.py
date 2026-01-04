"""
Text Generation Pipeline - Main Execution Script
Orchestrates the video text enrichment pipeline.

This script runs the complete pipeline:
1. Extract frames from videos
2. Extract CLIP features from frames
3. Segment videos using KTS
4. Generate captions for segments using BLIP-2
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import time
import csv
import gc
import torch

# Import pipeline modules
from generators.frame_extractor import FrameExtractor
from generators.feature_extractor import CLIPFeatureExtractor
from generators.kts_segmentor import KTSSegmentor
from generators.caption_generator import CaptionGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_train_split(dataset_name, dataset_root='datasets'):
    """
    Load training split video IDs for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MSRVTT', 'MSVD')
        dataset_root: Root directory containing datasets (default: 'datasets')
        
    Returns:
        Set of video IDs in the training split, or None if split file not found
    """
    dataset_path = Path(dataset_root) / dataset_name
    train_video_ids = set()
    
    if dataset_name.upper() == 'MSRVTT':
        # Read MSRVTT_train.9k.csv
        split_file = dataset_path / 'MSRVTT_train.9k.csv'
        
        if not split_file.exists():
            logger.warning(f"Training split file not found: {split_file}")
            logger.warning("Will process all videos in the dataset")
            return None
        
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'video_id' in row:
                        train_video_ids.add(row['video_id'])
            
            logger.info(f"Loaded {len(train_video_ids)} training videos from {split_file}")
            return train_video_ids
            
        except Exception as e:
            logger.error(f"Error reading MSRVTT split file: {str(e)}")
            return None
    
    elif dataset_name.upper() == 'MSVD':
        # Read train_list.txt
        split_file = dataset_path / 'train_list.txt'
        
        if not split_file.exists():
            logger.warning(f"Training split file not found: {split_file}")
            logger.warning("Will process all videos in the dataset")
            return None
        
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                for line in f:
                    video_name = line.strip()
                    if video_name:
                        train_video_ids.add(video_name)
            
            logger.info(f"Loaded {len(train_video_ids)} training videos from {split_file}")
            return train_video_ids
            
        except Exception as e:
            logger.error(f"Error reading MSVD split file: {str(e)}")
            return None
    
    else:
        logger.warning(f"No training split defined for dataset: {dataset_name}")
        logger.warning("Will process all videos in the dataset")
        return None


class TextEnrichmentPipeline:
    """Complete pipeline for video text enrichment."""
    
    def __init__(self, dataset_name, dataset_root='datasets', device='cuda', 
                 pretrained_dir='./pretrained'):
        """
        Initialize the pipeline.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'MSRVTT')
            dataset_root: Root directory containing datasets (default: 'datasets')
            device: Device to run inference on
            pretrained_dir: Directory to cache pretrained models
        """
        self.dataset_name = dataset_name
        self.dataset_root = Path(dataset_root)
        self.dataset_path = self.dataset_root / dataset_name
        self.device = device
        self.pretrained_dir = Path(pretrained_dir)
        
        # Create pretrained directory
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing pipeline for dataset: {dataset_name}")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Device: {device}")
    
    def step1_extract_frames(self, num_frames=12, target_size=(224, 224), skip_existing=True, whitelist_video_ids=None):
        """
        Step 1: Extract frames from videos.
        
        Args:
            num_frames: Number of frames to extract evenly from each video
            target_size: Target frame size
            skip_existing: Skip videos with existing frames
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Processing statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 1/4: Extracting Frames from Videos")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            extractor = FrameExtractor(
                self.dataset_path,
                target_size=target_size,
                num_frames=num_frames
            )
            
            stats = extractor.process_dataset(skip_existing=skip_existing, whitelist_video_ids=whitelist_video_ids)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Step 1 completed in {elapsed_time:.2f}s")
            logger.info(f"Processed: {stats['processed']}, "
                       f"Skipped: {stats['skipped']}, "
                       f"Failed: {stats['failed']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in Step 1: {str(e)}")
            raise
    
    def step2_extract_features(self, skip_existing=True, whitelist_video_ids=None):
        """
        Step 2: Extract CLIP features from frames.
        
        Args:
            skip_existing: Skip videos with existing features
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Processing statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 2/4: Extracting CLIP Features from Frames")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            extractor = CLIPFeatureExtractor(
                self.dataset_path,
                device=self.device
            )
            
            stats = extractor.process_dataset(skip_existing=skip_existing, whitelist_video_ids=whitelist_video_ids)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Step 2 completed in {elapsed_time:.2f}s")
            logger.info(f"Processed: {stats['processed']}, "
                       f"Skipped: {stats['skipped']}, "
                       f"Failed: {stats['failed']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in Step 2: {str(e)}")
            raise
    
    def step3_segment_videos(self, num_segments=None, penalty_coef=1.0, skip_existing=True, whitelist_video_ids=None):
        """
        Step 3: Segment videos using KTS.
        
        Args:
            num_segments: Target number of segments (None for automatic)
            penalty_coef: Penalty coefficient for model selection (default: 1.0)
            skip_existing: Skip videos with existing segments
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Processing statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 3/4: Segmenting Videos using KTS")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            segmentor = KTSSegmentor(
                self.dataset_path,
                num_segments=num_segments,
                penalty_coef=penalty_coef
            )
            
            stats = segmentor.process_dataset(skip_existing=skip_existing, whitelist_video_ids=whitelist_video_ids)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Step 3 completed in {elapsed_time:.2f}s")
            logger.info(f"Processed: {stats['processed']}, "
                       f"Skipped: {stats['skipped']}, "
                       f"Failed: {stats['failed']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in Step 3: {str(e)}")
            raise
    
    def step4_generate_captions(self, output_file='enriched_captions.json', whitelist_video_ids=None):
        """
        Step 4: Generate captions for video segments.
        
        Args:
            output_file: Name of output JSON file
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Processing statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 4/4: Generating Captions using BLIP-2")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            generator = CaptionGenerator(
                self.dataset_path,
                pretrained_dir=str(self.pretrained_dir),
                device=self.device,
                dataset_name=self.dataset_name
            )
            
            captions, stats = generator.process_dataset(output_file=output_file, whitelist_video_ids=whitelist_video_ids)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Step 4 completed in {elapsed_time:.2f}s")
            logger.info(f"Processed: {stats['processed']}, "
                       f"Failed: {stats['failed']}, "
                       f"Filtered: {stats.get('filtered', 0)}")
            logger.info(f"Generated captions for {len(captions)} videos")
            logger.info(f"Total captions: {sum(len(caps) for caps in captions.values())}")
            
            # Return stats for consistency with other steps
            return stats
            
        except Exception as e:
            logger.error(f"Error in Step 4: {str(e)}")
            raise
    
    def run_full_pipeline(self, num_frames=12, target_size=(224, 224), 
                         num_segments=None, penalty_coef=1.0,
                         skip_existing=True, output_file='enriched_captions.json',
                         train_split_only=True):
        """
        Run the complete text enrichment pipeline.
        
        Args:
            num_frames: Number of frames to extract evenly from each video
            target_size: Target frame size
            num_segments: Target number of segments per video
            penalty_coef: Penalty coefficient for model selection (default: 1.0)
            skip_existing: Skip already processed videos
            output_file: Name of output caption file
            train_split_only: Only process training split videos (default: True)
            
        Returns:
            Dictionary with results from each step
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING VIDEO TEXT ENRICHMENT PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Configuration:")
        logger.info(f"  - Num frames per video: {num_frames}")
        logger.info(f"  - Frame size: {target_size}")
        logger.info(f"  - Num segments: {num_segments or 'automatic'}")
        logger.info(f"  - KTS penalty_coef: {penalty_coef}")
        logger.info(f"  - Skip existing: {skip_existing}")
        logger.info(f"  - Train split only: {train_split_only}")
        logger.info(f"  - Device: {self.device}")
        
        # Load training split if needed
        train_video_ids = None
        if train_split_only:
            train_video_ids = load_train_split(self.dataset_name, self.dataset_root)
            if train_video_ids:
                logger.info(f"  - Processing {len(train_video_ids)} training videos only")
            else:
                logger.warning("  - Train split not found, processing all videos")
        
        logger.info("=" * 70 + "\n")
        
        pipeline_start_time = time.time()
        results = {}
        
        try:
            # Step 1: Extract frames
            results['step1'] = self.step1_extract_frames(
                num_frames=num_frames,
                target_size=target_size,
                skip_existing=skip_existing,
                whitelist_video_ids=train_video_ids
            )
            
            # Step 2: Extract features
            results['step2'] = self.step2_extract_features(
                skip_existing=skip_existing,
                whitelist_video_ids=train_video_ids
            )
            
            # Clean up GPU memory after CLIP feature extraction
            logger.info("Cleaning up GPU memory after Feature Extraction...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GPU memory cleanup complete")
            
            # Step 3: Segment videos
            results['step3'] = self.step3_segment_videos(
                num_segments=num_segments,
                penalty_coef=penalty_coef,
                skip_existing=skip_existing,
                whitelist_video_ids=train_video_ids
            )
            
            # Step 4: Generate captions
            results['step4'] = self.step4_generate_captions(
                output_file=output_file,
                whitelist_video_ids=train_video_ids
            )
            
            # Pipeline complete
            total_time = time.time() - pipeline_start_time
            
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            logger.info(f"Output saved to: {self.dataset_path / output_file}")
            logger.info("=" * 70 + "\n")
            
            return results
            
        except Exception as e:
            logger.error(f"\n{'='*70}")
            logger.error("PIPELINE FAILED")
            logger.error(f"{'='*70}")
            logger.error(f"Error: {str(e)}")
            logger.error(f"{'='*70}\n")
            raise


def main():
    """Main entry point for the text generation pipeline."""
    parser = argparse.ArgumentParser(
        description='Video Text Enrichment Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='MSRVTT',
                       help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='datasets',
                       help='Root directory containing datasets')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained',
                       help='Directory to cache pretrained models')
    
    # Frame extraction arguments
    parser.add_argument('--num_frames', type=int, default=12,
                       help='Number of frames to extract evenly from each video')
    parser.add_argument('--frame_size', type=int, default=224,
                       help='Target frame size (square)')
    
    # Segmentation arguments
    parser.add_argument('--num_segments', type=int, default=None,
                       help='Target number of segments per video (None for automatic)')
    parser.add_argument('--penalty_coef', type=float, default=1.0,
                       help='Penalty coefficient for KTS model selection (default: 1.0)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='enriched_captions.json',
                       help='Output filename for enriched captions')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip already processed videos')
    parser.add_argument('--no_skip_existing', action='store_false', dest='skip_existing',
                       help='Reprocess all videos')
    
    # Training split control
    parser.add_argument('--train_split_only', action='store_true', default=True,
                       help='Only process training split videos (default: True)')
    parser.add_argument('--process_all', action='store_false', dest='train_split_only',
                       help='Process all videos (not just training split)')
    
    # Pipeline control
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', 'all'],
                       default='all',
                       help='Which step to run (1-4 or all)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TextEnrichmentPipeline(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        device=args.device,
        pretrained_dir=args.pretrained_dir
    )
    
    # Load training split if needed
    train_video_ids = None
    if args.train_split_only:
        train_video_ids = load_train_split(args.dataset_name, args.dataset_root)
        if train_video_ids:
            logger.info(f"Processing {len(train_video_ids)} training videos only")
        else:
            logger.warning("Train split not found, processing all videos")
    
    # Run selected step(s)
    try:
        if args.step == 'all':
            # Run full pipeline
            pipeline.run_full_pipeline(
                num_frames=args.num_frames,
                target_size=(args.frame_size, args.frame_size),
                num_segments=args.num_segments,
                penalty_coef=args.penalty_coef,
                skip_existing=args.skip_existing,
                output_file=args.output,
                train_split_only=args.train_split_only
            )
        else:
            # Run individual step
            step_num = int(args.step)
            
            if step_num == 1:
                pipeline.step1_extract_frames(
                    num_frames=args.num_frames,
                    target_size=(args.frame_size, args.frame_size),
                    skip_existing=args.skip_existing,
                    whitelist_video_ids=train_video_ids
                )
            elif step_num == 2:
                pipeline.step2_extract_features(
                    skip_existing=args.skip_existing,
                    whitelist_video_ids=train_video_ids
                )
            elif step_num == 3:
                pipeline.step3_segment_videos(
                    num_segments=args.num_segments,
                    penalty_coef=args.penalty_coef,
                    skip_existing=args.skip_existing,
                    whitelist_video_ids=train_video_ids
                )
            elif step_num == 4:
                pipeline.step4_generate_captions(
                    output_file=args.output,
                    whitelist_video_ids=train_video_ids
                )
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
