"""
Generators Module
Contains all generator modules for video text enrichment pipeline.
"""

from .frame_extractor import FrameExtractor, extract_frames
from .feature_extractor import CLIPFeatureExtractor, extract_features
from .kts_segmentor import KTSSegmentor, segment_videos
from .caption_generator import CaptionGenerator, generate_captions

__all__ = [
    'FrameExtractor',
    'extract_frames',
    'CLIPFeatureExtractor',
    'extract_features',
    'KTSSegmentor',
    'segment_videos',
    'CaptionGenerator',
    'generate_captions',
]

__version__ = '1.0.0'
