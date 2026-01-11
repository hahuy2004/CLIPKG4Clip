"""
Farthest Query Selection (FQS) Algorithm
Phase 2: Select k most diverse queries from enriched set
"""

import numpy as np
import torch


def compute_distance_matrix(query_embeddings):
    """
    Compute pairwise distance matrix between query embeddings.
    
    Args:
        query_embeddings: torch.Tensor of shape (n_queries, embed_dim)
        
    Returns:
        np.ndarray: Distance matrix of shape (n_queries, n_queries)
    """
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.cpu().numpy()
    
    # Normalize embeddings
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized = query_embeddings / (norms + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(normalized, normalized.T)
    
    # Convert to distance (1 - cosine similarity)
    distance = 1 - similarity
    
    return distance


def farthest_query_selection(query_embeddings, k=2, return_indices=True):
    """
    Farthest Query Selection (FQS) algorithm.
    Selects k most diverse queries from enriched set using farthest point sampling.
    
    Algorithm:
    1. Initialize with original query (index 0)
    2. Iteratively select the query that is farthest from all selected queries
    3. Return k selected queries + original = k+1 total
    
    Args:
        query_embeddings: torch.Tensor or np.ndarray of shape (n_queries, embed_dim)
                         First embedding should be the original query
        k: Number of enriched queries to select (default: 2)
           Total selected = k + 1 (including original)
        return_indices: If True, return indices; else return embeddings
        
    Returns:
        If return_indices=True: list of selected indices [0, idx1, idx2, ...]
        If return_indices=False: selected embeddings array
    """
    if isinstance(query_embeddings, torch.Tensor):
        embeddings_np = query_embeddings.cpu().numpy()
    else:
        embeddings_np = query_embeddings.copy()
    
    n_queries = embeddings_np.shape[0]
    
    # Must have at least k+1 queries (1 original + k enriched)
    if n_queries < k + 1:
        raise ValueError(f"Need at least {k+1} queries, but got {n_queries}")
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(embeddings_np)
    
    # Initialize with original query (index 0)
    selected_indices = [0]
    remaining_indices = list(range(1, n_queries))
    
    # Iteratively select k more queries
    for _ in range(k):
        max_min_distance = -1
        farthest_idx = -1
        
        # For each remaining query, find minimum distance to selected set
        for idx in remaining_indices:
            # Distance from this query to all selected queries
            distances_to_selected = [distance_matrix[idx, s_idx] for s_idx in selected_indices]
            min_distance = min(distances_to_selected)
            
            # Select the query with maximum minimum distance (farthest point)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = idx
        
        # Add farthest query to selected set
        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)
    
    if return_indices:
        return selected_indices
    else:
        return embeddings_np[selected_indices]


def batch_farthest_query_selection(batch_query_embeddings, k=2):
    """
    Apply FQS to a batch of query sets.
    
    Args:
        batch_query_embeddings: torch.Tensor of shape (batch_size, n_queries, embed_dim)
        k: Number of enriched queries to select per sample
        
    Returns:
        list of lists: [[selected_indices for sample 1], [selected_indices for sample 2], ...]
    """
    batch_size = batch_query_embeddings.shape[0]
    batch_selected = []
    
    for i in range(batch_size):
        selected = farthest_query_selection(batch_query_embeddings[i], k=k, return_indices=True)
        batch_selected.append(selected)
    
    return batch_selected


def select_queries_by_indices(query_embeddings, selected_indices):
    """
    Extract selected query embeddings by indices.
    
    Args:
        query_embeddings: torch.Tensor of shape (n_queries, embed_dim) or (batch, n_queries, embed_dim)
        selected_indices: list of indices or list of lists for batch
        
    Returns:
        Selected embeddings
    """
    if isinstance(selected_indices[0], list):
        # Batch mode
        selected = []
        for i, indices in enumerate(selected_indices):
            selected.append(query_embeddings[i, indices])
        return torch.stack(selected)
    else:
        # Single sample
        return query_embeddings[selected_indices]


if __name__ == "__main__":
    # Test FQS algorithm
    print("Testing Farthest Query Selection Algorithm...")
    
    # Create dummy embeddings: 11 queries (1 original + 10 variations)
    np.random.seed(42)
    n_queries = 11
    embed_dim = 512
    
    # Original query
    original = np.random.randn(1, embed_dim)
    original = original / np.linalg.norm(original)
    
    # Generate variations with some noise
    variations = []
    for i in range(10):
        var = original + np.random.randn(1, embed_dim) * 0.3
        var = var / np.linalg.norm(var)
        variations.append(var)
    
    query_embeddings = np.vstack([original] + variations)
    
    # Apply FQS with k=2
    selected_indices = farthest_query_selection(query_embeddings, k=2)
    
    print(f"Original query index: 0")
    print(f"Selected indices (total {len(selected_indices)}): {selected_indices}")
    print(f"Expected: [0, idx1, idx2] where idx1, idx2 are from 1-10")
    
    # Verify distances
    distance_matrix = compute_distance_matrix(query_embeddings)
    print(f"\nDistances between selected queries:")
    for i, idx_i in enumerate(selected_indices):
        for j, idx_j in enumerate(selected_indices):
            if i < j:
                print(f"  Query {idx_i} <-> Query {idx_j}: {distance_matrix[idx_i, idx_j]:.4f}")
    
    print("\nFQS test completed successfully!")
