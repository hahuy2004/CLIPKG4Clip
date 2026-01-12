"""
Aggregator for combining multiple similarity matrices from enriched queries.

Two strategies:
1. Average Similarity: Simple average of similarity matrices
2. Majority Voting: Voting based on inverse rank scores
"""

import numpy as np


class Aggregator:
    """
    Aggregates multiple similarity matrices using different strategies.
    
    Strategies:
        1 = Majority Voting (voting based on inverse rank scores)
        2 = Average Similarity (simple average of similarity matrices)
    """
    
    def __init__(self, strategy=1):
        """
        Initialize Aggregator.
        
        Args:
            strategy (int): Aggregation strategy
                           1 = Majority Voting
                           2 = Average Similarity
        """
        if strategy not in [1, 2]:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 1 (Voting) or 2 (Average)")
        
        self.strategy = strategy
        self.strategy_name = "Majority Voting" if strategy == 1 else "Average Similarity"
    
    def aggregate(self, sim_matrices):
        """
        Aggregate multiple similarity matrices.
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
                                      k+1 similarity matrices to aggregate
        
        Returns:
            np.ndarray: Aggregated similarity matrix, shape (n_queries, n_videos)
        """
        if self.strategy == 1:
            return self.voting_aggregation(sim_matrices)
        else:  # strategy == 2
            return self.average_aggregation(sim_matrices)
    
    def average_aggregation(self, sim_matrices):
        """
        Average Similarity aggregation.
        
        Simply computes the arithmetic mean of all similarity matrices.
        
        Formula:
            Sim_final = (Sim_0 + Sim_1 + ... + Sim_k) / (k + 1)
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
        
        Returns:
            np.ndarray: Averaged similarity matrix, shape (n_queries, n_videos)
        """
        # Simple arithmetic mean along axis 0
        avg_sim = np.mean(sim_matrices, axis=0)
        return avg_sim
    
    def voting_aggregation(self, sim_matrices):
        """
        Majority Voting aggregation using inverse rank scores.
        
        Process:
        1. For each similarity matrix, convert to rankings
        2. Compute inverse rank score for each video
        3. Sum scores across all queries
        4. Create final similarity matrix based on aggregated scores
        
        Formula for each video v across all queries:
            Score(v) = Σ (1 / Rank_i(v))
        where Rank_i(v) is the rank of video v in the i-th query's ranking
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
        
        Returns:
            np.ndarray: Aggregated similarity matrix based on voting scores,
                       shape (n_queries, n_videos)
        """
        k_plus_1, n_queries, n_videos = sim_matrices.shape
        
        # Initialize aggregated similarity matrix
        final_sim = np.zeros((n_queries, n_videos))
        
        # Process each query independently
        for q_idx in range(n_queries):
            # Collect similarity scores for this query from all k+1 retrieval runs
            query_sims = sim_matrices[:, q_idx, :]  # (k+1, n_videos)
            
            # Initialize voting scores for each video
            video_scores = np.zeros(n_videos)
            
            # For each retrieval run (original + k enriched queries)
            for k_idx in range(k_plus_1):
                # Get similarity scores for this specific query-retrieval pair
                sim_scores = query_sims[k_idx]  # (n_videos,)
                
                # Convert to rankings: higher similarity = better rank (lower rank number)
                # argsort gives indices that would sort the array
                # [::-1] reverses to get descending order (best first)
                ranking = np.argsort(sim_scores)[::-1]
                
                # Create rank array: rank[video_idx] = rank of that video (1-indexed)
                ranks = np.zeros(n_videos, dtype=np.float32)
                for rank_position, video_idx in enumerate(ranking):
                    ranks[video_idx] = rank_position + 1  # 1-indexed rank
                
                # Compute inverse rank scores: 1 / rank
                inverse_ranks = 1.0 / ranks
                
                # Accumulate scores
                video_scores += inverse_ranks
            
            # Normalize scores to [0, 1] for similarity compatibility
            if video_scores.max() > 0:
                video_scores = video_scores / video_scores.max()
            
            # Assign aggregated scores as final similarity for this query
            final_sim[q_idx, :] = video_scores
        
        return final_sim


def test_aggregator():
    """Test the Aggregator class with dummy data."""
    print("="*70)
    print("TESTING AGGREGATOR")
    print("="*70)
    
    np.random.seed(42)
    
    # Create dummy similarity matrices
    k_plus_1 = 3  # 1 original + 2 enriched = 3 total queries per video
    n_queries = 10  # 10 test queries
    n_videos = 100  # 100 candidate videos
    
    # Generate k+1 similarity matrices
    sim_matrices = np.random.rand(k_plus_1, n_queries, n_videos)
    
    print(f"\nInput: {k_plus_1} similarity matrices")
    print(f"Shape: ({k_plus_1}, {n_queries}, {n_videos})")
    print(f"  - {k_plus_1} retrieval runs (1 original + {k_plus_1-1} enriched)")
    print(f"  - {n_queries} test queries")
    print(f"  - {n_videos} candidate videos")
    
    # Test Strategy 1: Majority Voting
    print("\n" + "-"*70)
    print("Testing Strategy 1: Majority Voting")
    print("-"*70)
    
    aggregator_voting = Aggregator(strategy=1)
    final_sim_voting = aggregator_voting.aggregate(sim_matrices)
    
    print(f"Output shape: {final_sim_voting.shape}")
    print(f"Output range: [{final_sim_voting.min():.4f}, {final_sim_voting.max():.4f}]")
    
    # Show top-5 videos for first query
    query_0_scores = final_sim_voting[0]
    top5_indices = np.argsort(query_0_scores)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (Voting):")
    for rank, video_idx in enumerate(top5_indices, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores[video_idx]:.4f})")
    
    # Test Strategy 2: Average Similarity
    print("\n" + "-"*70)
    print("Testing Strategy 2: Average Similarity")
    print("-"*70)
    
    aggregator_avg = Aggregator(strategy=2)
    final_sim_avg = aggregator_avg.aggregate(sim_matrices)
    
    print(f"Output shape: {final_sim_avg.shape}")
    print(f"Output range: [{final_sim_avg.min():.4f}, {final_sim_avg.max():.4f}]")
    
    # Show top-5 videos for first query
    query_0_scores_avg = final_sim_avg[0]
    top5_indices_avg = np.argsort(query_0_scores_avg)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (Average):")
    for rank, video_idx in enumerate(top5_indices_avg, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores_avg[video_idx]:.4f})")
    
    # Compare strategies
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nDifference between strategies:")
    diff = np.abs(final_sim_voting - final_sim_avg)
    print(f"  Mean absolute difference: {diff.mean():.4f}")
    print(f"  Max absolute difference: {diff.max():.4f}")
    
    # Agreement on top-1
    agreement_count = 0
    for q_idx in range(n_queries):
        top1_voting = np.argmax(final_sim_voting[q_idx])
        top1_avg = np.argmax(final_sim_avg[q_idx])
        if top1_voting == top1_avg:
            agreement_count += 1
    
    agreement_rate = 100.0 * agreement_count / n_queries
    print(f"\nTop-1 agreement: {agreement_count}/{n_queries} ({agreement_rate:.1f}%)")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    test_aggregator()
