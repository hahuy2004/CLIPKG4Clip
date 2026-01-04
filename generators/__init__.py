"""
Generators Module
Contains all generator modules for video text enrichment pipeline.
"""

# Lazy imports to avoid loading heavy dependencies unnecessarily
# Import only when needed to prevent TensorFlow/transformers conflicts

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

def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies."""
    if name == 'FrameExtractor':
        from .frame_extractor import FrameExtractor
        return FrameExtractor
    elif name == 'extract_frames':
        from .frame_extractor import extract_frames
        return extract_frames
    elif name == 'CLIPFeatureExtractor':
        from .feature_extractor import CLIPFeatureExtractor
        return CLIPFeatureExtractor
    elif name == 'extract_features':
        from .feature_extractor import extract_features
        return extract_features
    elif name == 'KTSSegmentor':
        from .kts_segmentor import KTSSegmentor
        return KTSSegmentor
    elif name == 'segment_videos':
        from .kts_segmentor import segment_videos
        return segment_videos
    elif name == 'CaptionGenerator':
        from .caption_generator import CaptionGenerator
        return CaptionGenerator
    elif name == 'generate_captions':
        from .caption_generator import generate_captions
        return generate_captions
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = '1.0.0'
