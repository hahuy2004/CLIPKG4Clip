"""
Majority Voting Aggregation using Reciprocal Rank Fusion
Phase 2: Aggregate retrieval results from multiple queries
"""

import numpy as np
import torch
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.
    
    Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
    where rank_i(d) is the rank of document d in list i
    
    Args:
        ranked_lists: List of ranked lists, each containing video indices
                     e.g., [[vid1, vid2, ...], [vid3, vid1, ...], ...]
        k: Constant for RRF (default: 60, commonly used value)
        
    Returns:
        np.ndarray: Final ranking of video indices sorted by RRF score
        dict: Video index -> RRF score mapping
    """
    rrf_scores = defaultdict(float)
    
    # Accumulate scores from each ranked list
    for ranked_list in ranked_lists:
        for rank, video_idx in enumerate(ranked_list):
            # rank is 0-indexed, so rank+1 for 1-indexed
            rrf_scores[video_idx] += 1.0 / (k + rank + 1)
    
    # Sort by RRF score (descending)
    sorted_videos = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_ranking = np.array([video_idx for video_idx, score in sorted_videos])
    
    return final_ranking, dict(sorted_videos)


def borda_count(ranked_lists, n_candidates):
    """
    Borda Count for combining multiple ranked lists.
    Each video gets points based on its position: (n - rank) points
    
    Args:
        ranked_lists: List of ranked lists containing video indices
        n_candidates: Total number of candidate videos
        
    Returns:
        np.ndarray: Final ranking of video indices sorted by Borda count
        dict: Video index -> Borda count mapping
    """
    borda_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, video_idx in enumerate(ranked_list):
            # Assign points: higher rank = more points
            points = n_candidates - rank
            borda_scores[video_idx] += points
    
    # Sort by Borda count (descending)
    sorted_videos = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    final_ranking = np.array([video_idx for video_idx, score in sorted_videos])
    
    return final_ranking, dict(sorted_videos)


def majority_voting_aggregation(similarity_matrices, top_k=100, method='rrf', rrf_k=60):
    """
    Aggregate multiple similarity matrices using majority voting.
    
    Args:
        similarity_matrices: List of similarity matrices from different queries
                            Each matrix shape: (n_queries, n_videos)
                            Or np.ndarray of shape (n_selected_queries, n_queries, n_videos)
        top_k: Number of top candidates to consider from each query (default: 100)
        method: Aggregation method - 'rrf' or 'borda' (default: 'rrf')
        rrf_k: Constant for RRF method (default: 60)
        
    Returns:
        np.ndarray: Final aggregated similarity matrix (n_queries, n_videos)
    """
    if isinstance(similarity_matrices, np.ndarray):
        if similarity_matrices.ndim == 3:
            # Shape: (n_selected_queries, n_queries, n_videos)
            n_selected, n_queries, n_videos = similarity_matrices.shape
            matrices_list = [similarity_matrices[i] for i in range(n_selected)]
        else:
            matrices_list = [similarity_matrices]
    else:
        matrices_list = similarity_matrices
    
    n_queries = matrices_list[0].shape[0]
    n_videos = matrices_list[0].shape[1]
    
    # Initialize aggregated similarity matrix
    aggregated_sim = np.zeros((n_queries, n_videos))
    
    # Process each query independently
    for q_idx in range(n_queries):
        # Get ranked lists from each selected query
        ranked_lists = []
        
        for sim_matrix in matrices_list:
            # Get similarity scores for this query
            sim_scores = sim_matrix[q_idx]
            
            # Get top-k video indices sorted by similarity (descending)
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            ranked_lists.append(top_indices)
        
        # Aggregate rankings
        if method == 'rrf':
            final_ranking, scores_dict = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
        elif method == 'borda':
            final_ranking, scores_dict = borda_count(ranked_lists, n_videos)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Convert scores back to similarity values
        # Normalize scores to [0, 1] range for compatibility
        if scores_dict:
            max_score = max(scores_dict.values())
            min_score = min(scores_dict.values())
            score_range = max_score - min_score
            
            if score_range > 0:
                for video_idx, score in scores_dict.items():
                    normalized_score = (score - min_score) / score_range
                    aggregated_sim[q_idx, video_idx] = normalized_score
            else:
                # All scores are the same
                for video_idx in scores_dict.keys():
                    aggregated_sim[q_idx, video_idx] = 1.0
    
    return aggregated_sim


def batch_majority_voting(batch_similarity_matrices, selected_indices_batch, top_k=100, method='rrf'):
    """
    Apply majority voting to a batch of samples.
    
    Args:
        batch_similarity_matrices: torch.Tensor or np.ndarray 
                                  shape: (batch_size, n_queries, n_videos)
        selected_indices_batch: List of selected query indices for each sample
                               e.g., [[0, 3, 7], [0, 2, 5], ...]
        top_k: Number of top candidates to consider
        method: Aggregation method
        
    Returns:
        Aggregated similarity matrices: np.ndarray (batch_size, n_queries, n_videos)
    """
    if isinstance(batch_similarity_matrices, torch.Tensor):
        batch_similarity_matrices = batch_similarity_matrices.cpu().numpy()
    
    batch_size = batch_similarity_matrices.shape[0]
    batch_aggregated = []
    
    for i in range(batch_size):
        # Get selected query similarities for this sample
        selected_indices = selected_indices_batch[i]
        selected_sim_matrices = batch_similarity_matrices[i, selected_indices, :]
        
        # Aggregate
        aggregated = majority_voting_aggregation(
            [selected_sim_matrices[j:j+1] for j in range(len(selected_indices))],
            top_k=top_k,
            method=method
        )
        
        batch_aggregated.append(aggregated)
    
    return np.stack(batch_aggregated)


if __name__ == "__main__":
    # Test majority voting aggregation
    print("Testing Majority Voting Aggregation...")
    
    np.random.seed(42)
    
    # Create dummy similarity matrices
    n_queries = 10
    n_videos = 100
    n_selected_queries = 3  # Original + 2 enriched
    
    # Simulate 3 similarity matrices from selected queries
    sim_matrices = []
    for i in range(n_selected_queries):
        sim = np.random.rand(n_queries, n_videos)
        sim_matrices.append(sim)
    
    print(f"Input: {n_selected_queries} similarity matrices, each shape {sim_matrices[0].shape}")
    
    # Test RRF aggregation
    aggregated_rrf = majority_voting_aggregation(sim_matrices, top_k=50, method='rrf')
    print(f"RRF aggregated matrix shape: {aggregated_rrf.shape}")
    
    # Test Borda Count aggregation
    aggregated_borda = majority_voting_aggregation(sim_matrices, top_k=50, method='borda')
    print(f"Borda Count aggregated matrix shape: {aggregated_borda.shape}")
    
    # Verify top-1 retrieval for first query
    query_0_top1_rrf = np.argmax(aggregated_rrf[0])
    query_0_top1_borda = np.argmax(aggregated_borda[0])
    
    print(f"\nFor query 0:")
    print(f"  RRF Top-1 video: {query_0_top1_rrf}")
    print(f"  Borda Top-1 video: {query_0_top1_borda}")
    
    print("\nMajority voting test completed successfully!")
