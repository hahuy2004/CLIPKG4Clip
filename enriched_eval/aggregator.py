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
        Improved Majority Voting aggregation using Weighted Reciprocal Rank Fusion (RRF).
        
        Key Improvements over naive 1/rank:
        1. Weighted voting: Original query has higher weight than enriched queries
        2. Smoothing constant k: Reduces harshness of pure 1/rank formula
        3. No normalization: Preserves relative magnitudes across queries
        
        Process:
        1. Assign weights: Original query = 1.0, Enriched queries = 0.4
        2. For each similarity matrix, convert to rankings (0-based)
        3. Apply weighted RRF formula: w * (1 / (k + rank + 1))
        4. Sum weighted scores across all query variants
        
        Formula:
            Score(q, v) = Σ_i [ w_i * (1 / (k + Rank_i(q, v) + 1)) ]
        where:
            w_i = weight of i-th query variant (original=1.0, enriched=0.4)
            k = smoothing constant (1.0)
            Rank_i(q, v) = 0-based rank of video v for query q in variant i
        
        Why this works:
        - Weighted: Original query (ground truth) has 2.5x influence vs enriched
        - Smoothing k=1.0: Rank 1→0.5, Rank 2→0.33 (33% drop vs 50% in pure 1/rank)
        - No normalization: Fair comparison across different queries
        
        Example (k=1.0, weights=[1.0, 0.4, 0.4]):
            Video A (correct):
                Original: Rank 0 → 1.0 * 1/(1+0+1) = 0.50
                Enriched1: Rank 2 → 0.4 * 1/(1+2+1) = 0.10
                Enriched2: Rank 2 → 0.4 * 1/(1+2+1) = 0.10
                Total: 0.70
            
            Video B (incorrect):
                Original: Rank 1 → 1.0 * 1/(1+1+1) = 0.33
                Enriched1: Rank 0 → 0.4 * 1/(1+0+1) = 0.20
                Enriched2: Rank 1 → 0.4 * 1/(1+1+1) = 0.13
                Total: 0.66
            
            → Video A wins! Original query's signal is preserved.
        
        Args:
            sim_matrices (np.ndarray): Shape (k+1, n_queries, n_videos)
                                      k+1 similarity matrices from query variants
        
        Returns:
            np.ndarray: Aggregated similarity matrix, shape (n_queries, n_videos)
        """
        k_plus_1, n_queries, n_videos = sim_matrices.shape
        
        # ---------------------------------------------------------
        # IMPROVED AGGREGATION LOGIC (Weighted RRF)
        # ---------------------------------------------------------
        
        # 1. Configure weights for query variants
        # Original query (index 0) is most important
        # Enriched queries (index > 0) provide supporting evidence
        weights = [1.0] + [0.4] * (k_plus_1 - 1)
        
        # 2. Smoothing constant for RRF
        # k=1.0 provides good balance:
        #   - Rank 1: 1/(1+1) = 0.50
        #   - Rank 2: 1/(1+2) = 0.33 (33% drop, gentler than 50%)
        #   - Rank 10: 1/(1+10) = 0.09
        # Smaller k → steeper curve (better R@1)
        # Larger k → flatter curve (better R@5, R@10)
        k_smooth = 1.0
        
        # Initialize final score matrix
        final_score_matrix = np.zeros((n_queries, n_videos))
        
        # 3. Process each query variant with its weight
        for idx, sim_matrix in enumerate(sim_matrices):
            w = weights[idx]
            
            # Get 0-based ranks for each query-video pair
            # argsort(-sim_matrix, axis=1): sort descending (highest sim first)
            # argsort again: convert sorted indices to ranks (0-based)
            # Result: ranks[i, j] = rank of video j for query i (0=best)
            ranks = np.argsort(np.argsort(-sim_matrix, axis=1), axis=1)
            
            # Apply weighted RRF formula: w * (1 / (k + rank + 1))
            # rank + 1: convert 0-based to 1-based for formula
            score = w * (1.0 / (k_smooth + ranks + 1))
            
            # Accumulate weighted scores
            final_score_matrix += score
        
        # 4. No normalization needed
        # compute_metrics() only cares about relative ranking order,
        # not absolute score magnitudes. Normalization can introduce bias
        # across different queries.
        
        return final_score_matrix


def test_aggregator():
    """Test the Aggregator class with dummy data and verify weighted RRF logic."""
    print("="*70)
    print("TESTING IMPROVED AGGREGATOR (Weighted RRF)")
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
    print(f"\nWeights: [1.0 (original), 0.4 (enriched1), 0.4 (enriched2)]")
    print(f"Smoothing constant k: 1.0")
    
    # Test Strategy 1: Weighted RRF Voting
    print("\n" + "-"*70)
    print("Testing Strategy 1: Weighted RRF Voting")
    print("-"*70)
    
    aggregator_voting = Aggregator(strategy=1)
    final_sim_voting = aggregator_voting.aggregate(sim_matrices)
    
    print(f"Output shape: {final_sim_voting.shape}")
    print(f"Output range: [{final_sim_voting.min():.4f}, {final_sim_voting.max():.4f}]")
    
    # Show top-5 videos for first query
    query_0_scores = final_sim_voting[0]
    top5_indices = np.argsort(query_0_scores)[::-1][:5]
    print(f"\nQuery 0 - Top 5 videos (Weighted RRF):")
    for rank, video_idx in enumerate(top5_indices, 1):
        print(f"  Rank {rank}: Video {video_idx:3d} (score: {query_0_scores[video_idx]:.4f})")
    
    # Verify weighted contribution
    print(f"\nVerifying weighted RRF formula for Query 0, Video {top5_indices[0]}:")
    vid_idx = top5_indices[0]
    for k_idx in range(k_plus_1):
        sim = sim_matrices[k_idx, 0, vid_idx]
        rank = np.sum(sim_matrices[k_idx, 0, :] > sim)  # 0-based rank
        weight = 1.0 if k_idx == 0 else 0.4
        contribution = weight * (1.0 / (1.0 + rank + 1))
        variant_name = "Original" if k_idx == 0 else f"Enriched{k_idx}"
        print(f"  {variant_name}: rank={rank}, weight={weight:.1f}, contribution={contribution:.4f}")
    
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
