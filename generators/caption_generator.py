"""
Caption Generator Module
Generates captions for video segments using BLIP-2.
"""
import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generate captions for video segments using BLIP-2."""
    
    def __init__(self, dataset_root, pretrained_dir='./pretrained', device='cuda'):
        """
        Initialize Caption Generator.
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'dataset/MSRVTT')
            pretrained_dir: Directory to cache pretrained models
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.dataset_root = Path(dataset_root)
        self.frames_dir = self.dataset_root / 'frames'
        self.segments_dir = self.dataset_root / 'segments'
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.pretrained_dir = Path(pretrained_dir)
        
        # Create pretrained directory
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        
        # Load BLIP-2 model
        logger.info(f"Loading BLIP-2 model on {self.device}")
        try:
            model_name = "Salesforce/blip2-opt-2.7b-coco"
            
            # Load processor and model with custom cache directory
            self.processor = Blip2Processor.from_pretrained(
                model_name,
                cache_dir=str(self.pretrained_dir)
            )
            
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=str(self.pretrained_dir),
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("BLIP-2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {str(e)}")
            raise
    
    def generate_caption(self, image_path):
        """
        Generate caption for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Generated caption string
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(
                self.device, 
                dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    temperature=1.0,
                    do_sample=False
                )
            
            # Decode caption
            caption = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return caption
            
        except Exception as e:
            logger.warning(f"Failed to generate caption for {image_path}: {str(e)}")
            return ""
    
    def caption_video(self, video_id):
        """
        Generate captions for all segments of a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of captions (one per segment)
        """
        # Load segments
        segments_path = self.segments_dir / f"{video_id}.json"
        if not segments_path.exists():
            logger.warning(f"Segments not found for {video_id}")
            return None
        
        try:
            with open(segments_path, 'r') as f:
                segments_data = json.load(f)
            
            segments = segments_data['segments']
            frames_dir = self.frames_dir / video_id
            
            if not frames_dir.exists():
                logger.warning(f"Frames directory not found for {video_id}")
                return None
            
            # Get all frame files
            frame_files = sorted(list(frames_dir.glob('*.jpg')) + 
                               list(frames_dir.glob('*.png')))
            
            if not frame_files:
                logger.warning(f"No frames found for {video_id}")
                return None
            
            captions = []
            
            # Generate caption for each segment
            for segment in segments:
                start_frame = segment['start_frame']
                end_frame = segment['end_frame']
                
                # Calculate middle frame index
                middle_frame_idx = (start_frame + end_frame) // 2
                
                # Ensure index is within bounds
                middle_frame_idx = min(middle_frame_idx, len(frame_files) - 1)
                
                # Get middle frame path
                middle_frame_path = frame_files[middle_frame_idx]
                
                # Generate caption
                caption = self.generate_caption(middle_frame_path)
                captions.append(caption)
                
                logger.debug(f"{video_id} segment [{start_frame}-{end_frame}]: {caption}")
            
            logger.info(f"Generated {len(captions)} captions for {video_id}")
            return captions
            
        except Exception as e:
            logger.error(f"Error captioning {video_id}: {str(e)}")
            return None
    
    def process_dataset(self, output_file='enriched_captions.json', whitelist_video_ids=None):
        """
        Process all videos in the dataset and generate enriched captions.
        
        Args:
            output_file: Name of output JSON file
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Dictionary mapping video_id to list of captions
        """
        if not self.segments_dir.exists():
            logger.error(f"Segments directory not found: {self.segments_dir}")
            return {}
        
        # Get all segment files
        segment_files = list(self.segments_dir.glob('*.json'))
        
        if not segment_files:
            logger.warning(f"No segment files found in {self.segments_dir}")
            return {}
        
        logger.info(f"Found {len(segment_files)} videos to caption")
        
        enriched_captions = {}
        stats = {'processed': 0, 'failed': 0, 'filtered': 0}
        
        # Process each video
        for segment_file in tqdm(segment_files, desc="Generating captions"):
            video_id = segment_file.stem
            
            # Filter by whitelist if provided
            if whitelist_video_ids is not None and video_id not in whitelist_video_ids:
                logger.debug(f"Filtering out {video_id} (not in training split)")
                stats['filtered'] += 1
                continue
            
            # Generate captions
            captions = self.caption_video(video_id)
            
            if captions is not None and len(captions) > 0:
                enriched_captions[video_id] = captions
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        # Save enriched captions
        output_path = self.dataset_root / output_file
        with open(output_path, 'w') as f:
            json.dump(enriched_captions, f, indent=2)
        
        logger.info(f"Caption generation complete. Processed: {stats['processed']}, "
                   f"Failed: {stats['failed']}, Filtered: {stats.get('filtered', 0)}")
        logger.info(f"Saved enriched captions to {output_path}")
        
        return enriched_captions


def generate_captions(dataset_name, dataset_root='dataset', pretrained_dir='./pretrained', 
                     device='cuda', output_file='enriched_captions.json'):
    """
    Convenience function to generate captions for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MSRVTT')
        dataset_root: Root directory containing datasets
        pretrained_dir: Directory to cache pretrained models
        device: Device to run on ('cuda' or 'cpu')
        output_file: Name of output JSON file
        
    Returns:
        Dictionary mapping video_id to list of captions
    """
    dataset_path = Path(dataset_root) / dataset_name
    generator = CaptionGenerator(dataset_path, pretrained_dir=pretrained_dir, device=device)
    return generator.process_dataset(output_file=output_file)


if __name__ == "__main__":
    # Test the caption generator
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate captions for video segments')
    parser.add_argument('--dataset_name', type=str, default='MSRVTT',
                       help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Root directory containing datasets')
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained',
                       help='Directory to cache pretrained models')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--output', type=str, default='enriched_captions.json',
                       help='Output filename')
    
    args = parser.parse_args()
    
    captions = generate_captions(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        pretrained_dir=args.pretrained_dir,
        device=args.device,
        output_file=args.output
    )
    
    print(f"\nCaption generation complete:")
    print(f"  Videos processed: {len(captions)}")
    print(f"  Total captions: {sum(len(caps) for caps in captions.values())}")
