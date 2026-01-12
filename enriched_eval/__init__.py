"""
Enriched Evaluation Module for Text-Video Retrieval
Based on ICLR 2025 Paper: Bridging Information Asymmetry
"""

from .query_generator import generate_enriched_queries
from .fqs_selector import farthest_query_selection
from .aggregator import Aggregator

# NOTE: enriched_dataloader is currently unused in main evaluation code
# Commented out to avoid import issues - uncomment if needed in future
# from .enriched_dataloader import create_enriched_dataloader

__all__ = [
    'generate_enriched_queries',
    'farthest_query_selection',
    'Aggregator',
    # 'create_enriched_dataloader'  # Commented - currently unused
]
