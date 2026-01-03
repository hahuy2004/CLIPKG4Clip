"""
Caption Generator Module
Generates captions for video segments using BLIP-2.
"""
import os
import json
import torch
import pickle
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generate captions for video segments using BLIP-2."""
    
    def __init__(self, dataset_root, pretrained_dir='./pretrained', device='cuda', dataset_name=None):
        """
        Initialize Caption Generator.
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'dataset/MSRVTT')
            pretrained_dir: Directory to cache pretrained models
            device: Device to run inference on ('cuda' or 'cpu')
            dataset_name: Name of dataset (for format detection: 'MSRVTT', 'MSVD', etc.)
        """
        self.dataset_root = Path(dataset_root)
        self.frames_dir = self.dataset_root / 'frames'
        self.segments_dir = self.dataset_root / 'segments'
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.pretrained_dir = Path(pretrained_dir)
        self.dataset_name = dataset_name
        
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
            
            logger.debug(f"Generated {len(captions)} captions for {video_id}")
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
        
        # Save enriched captions in appropriate format
        self._save_captions(enriched_captions, output_file)
        
        logger.info(f"Caption generation complete. Processed: {stats['processed']}, "
                   f"Failed: {stats['failed']}, Filtered: {stats.get('filtered', 0)}")
        
        return enriched_captions
    
    def _save_captions(self, enriched_captions, output_file):
        """
        Save captions in dataset-specific format.
        
        Args:
            enriched_captions: Dictionary of {video_id: [caption1, caption2, ...]}
            output_file: Output filename
        """
        # Detect dataset type
        dataset_type = None
        if self.dataset_name:
            dataset_type = self.dataset_name.upper()
        
        if dataset_type == 'MSRVTT':
            # MSRVTT format: JSON with sentences array
            self._save_msrvtt_format(enriched_captions, output_file)
        elif dataset_type == 'MSVD':
            # MSVD format: Pickle with word lists
            self._save_msvd_format(enriched_captions, output_file)
        else:
            # Default: Simple JSON format
            self._save_json_format(enriched_captions, output_file)
    
    def _save_json_format(self, enriched_captions, output_file):
        """Save as simple JSON format."""
        output_path = self.dataset_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_captions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved captions (JSON format) to {output_path}")
    
    def _save_msrvtt_format(self, enriched_captions, output_file):
        """
        Save in MSRVTT format compatible with dataloader.
        
        Format (matching MSRVTT_data.json structure):
        {
            "videos": [...]  # copied from original MSRVTT_data.json
            "sentences": [
                {"video_id": "video0", "caption": "..."},
                ...
            ]
        }
        """
        sentences = []
        for video_id, captions in enriched_captions.items():
            for caption in captions:
                sentences.append({
                    "video_id": video_id,
                    "caption": caption
                })
        
        # Load "videos" field from original MSRVTT_data.json
        # Path: datasets/MSRVTT/MSRVTT_data.json
        videos_data = []
        original_json_path = self.dataset_root / 'MSRVTT_data.json'
        
        logger.info(f"Looking for original MSRVTT_data.json at: {original_json_path.absolute()}")
        
        try:
            if original_json_path.exists():
                logger.info(f"Loading videos metadata from {original_json_path}")
                with open(original_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                    videos_data = original_data.get('videos', [])
                    logger.info(f"Successfully loaded {len(videos_data)} videos metadata from MSRVTT_data.json")
            else:
                logger.error(f"Original MSRVTT_data.json not found at {original_json_path.absolute()}")
                logger.warning("Using empty 'videos' array (may cause issues with parent-child video mapping)")
        except Exception as e:
            logger.error(f"Failed to load videos data from MSRVTT_data.json: {e}")
            logger.warning("Using empty 'videos' array")
        
        # Match original MSRVTT_data.json structure: videos first, then sentences
        output_data = {
            "videos": videos_data,
            "sentences": sentences
        }
        
        # Save as JSON
        output_path = self.dataset_root / output_file.replace('.json', '_enriched.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved captions (MSRVTT format) to {output_path}")
        
        # Also save simple format for reference
        simple_path = self.dataset_root / output_file
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_captions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved captions (simple format) to {simple_path}")
    
    def _save_msvd_format(self, enriched_captions, output_file):
        """
        Save in MSVD format compatible with dataloader.
        
        Format (pickle):
        {
            "video_id": [
                ["word1", "word2", "word3"],  # caption 1 as word list
                ["word4", "word5"],           # caption 2 as word list
                ...
            ]
        }
        """
        # Convert captions to word lists
        msvd_captions = {}
        for video_id, captions in enriched_captions.items():
            msvd_captions[video_id] = []
            for caption in captions:
                # Split caption into words
                words = caption.lower().split()
                msvd_captions[video_id].append(words)
        
        # Save as pickle file
        pickle_path = self.dataset_root / 'enriched-captions.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(msvd_captions, f)
        logger.info(f"Saved captions (MSVD pickle format) to {pickle_path}")
        
        # Also save simple JSON for reference
        json_path = self.dataset_root / output_file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_captions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved captions (simple format) to {json_path}")


def generate_captions(dataset_name, dataset_root='datasets', pretrained_dir='./pretrained', 
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
    generator = CaptionGenerator(dataset_path, pretrained_dir=pretrained_dir, device=device, dataset_name=dataset_name)
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
