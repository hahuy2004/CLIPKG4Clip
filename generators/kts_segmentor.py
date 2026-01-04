"""
KTS Segmentor Module
Performs Kernel Temporal Segmentation on video features to detect event boundaries.
"""
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def compute_change_points_kts(features, num_segments=None, penalty_coef=1.0):
    """
    Kernel Temporal Segmentation (KTS) - Algorithm 1 from Potapov et al. (2014).
    Exact implementation following the paper line-by-line.
    
    Args:
        features: Feature matrix of shape (N, feature_dim)
        num_segments: Target number of segments (if None, auto-select using penalty)
                     Note: number of segments = number of change points + 1
        penalty_coef: Penalty coefficient C for model selection (default: 1.0)
        
    Returns:
        List of change point indices [0, t1, t2, ..., t_m, N] where segments are [ti, ti+1)
    """
    N = features.shape[0]
    
    if N <= 1:
        return [0, N]
    
    # ============================================================
    # STEP 1: L2 Normalization (per frame)
    # Input: temporal sequence x_0, x_1, ..., x_{N-1}
    # ============================================================
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # ============================================================
    # STEP 2: Compute Gram Matrix (Linear Kernel) - Algorithm 1, Line 1
    # K[i,j] = K(x_i, x_j) = x_i^T @ x_j (cosine similarity after normalization)
    # Cost: O(N^2 * d), where d = feature_dim
    # ============================================================
    K = np.dot(features, features.T)  # Shape: (N, N)
    
    # ============================================================
    # STEP 3: Build 2D Summed-Area Table (Integral Image) - Algorithm 1, Line 2
    # Reference [34]: Viola & Jones, "Robust real-time object detection" (2001)
    # 
    # S[i,j] = sum of K[0:i, 0:j] (cumulative sum over 2D rectangular region)
    # This allows O(1) computation of any rectangular sub-matrix sum.
    # 
    # Construction: S[i,j] = K[i-1,j-1] + S[i-1,j] + S[i,j-1] - S[i-1,j-1]
    # Query (sum of K[r1:r2, c1:c2]): S[r2,c2] - S[r1,c2] - S[r2,c1] + S[r1,c1]
    # ============================================================
    S = np.zeros((N + 1, N + 1), dtype=np.float64)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            S[i, j] = K[i-1, j-1] + S[i-1, j] + S[i, j-1] - S[i-1, j-1]
    
    # ============================================================
    # STEP 4: Variance Function - Algorithm 1, Line 3
    # Unnormalized variance for segment [t, t+d):
    # v(t, t+d) = sum_{i=t}^{t+d-1} K_{ii} - (1/d) * sum_{i,j=t}^{t+d-1} K_{ij}
    # 
    # Using the 2D integral image, the second term is computed in O(1):
    # sum_{i,j=t}^{t+d-1} K_{ij} = S[t+d, t+d] - S[t, t+d] - S[t+d, t] + S[t, t]
    # ============================================================
    def compute_variance(t_start, t_end):
        """
        Compute unnormalized variance for segment [t_start, t_end) in O(1).
        """
        d = t_end - t_start
        if d <= 0:
            return 0.0
        
        # Sum of diagonal elements: sum_{i=t}^{t+d-1} K[i,i]
        diag_sum = np.sum(K[range(t_start, t_end), range(t_start, t_end)])
        
        # Sum of all elements in block K[t:t+d, t:t+d] using 4-corner formula
        block_sum = S[t_end, t_end] - S[t_start, t_end] - S[t_end, t_start] + S[t_start, t_start]
        
        # Variance formula from paper
        variance = diag_sum - (block_sum / d)
        return variance
    
    # ============================================================
    # STEP 5: Dynamic Programming - Algorithm 1, Line 4
    # L[i, j] = minimum cost for first j elements with i change-points
    # 
    # CRITICAL: i = number of change-points, NOT number of segments
    # With i change-points, you have (i+1) segments
    # 
    # Recurrence (Algorithm 1, Line 4):
    # L_{i,j} = min_{t=i,...,j-1} (L_{i-1,t} + v_{t,j})
    # 
    # Initialization (Algorithm 1, implied):
    # L_{0,j} = v_{0,j}  (0 change-points = 1 segment covering [0,j))
    # ============================================================
    
    # Determine maximum number of change-points to consider
    if num_segments is not None:
        # User specified number of segments -> number of change-points = segments - 1
        m_max = num_segments - 1
    else:
        # Auto-select: allow up to N-1 change-points (N segments)
        m_max = N - 1
    
    m_max = max(0, min(m_max, N - 1))  # Clamp to [0, N-1]
    
    # DP table: L[i, j] = minimum cost for j elements with i change-points
    L = np.full((m_max + 1, N + 1), np.inf, dtype=np.float64)
    
    # Backtracking table: stores the position of the i-th change-point for state (i, j)
    backtrack = np.zeros((m_max + 1, N + 1), dtype=np.int32)
    
    # Base case: 0 change-points (1 segment)
    # L[0, j] = v(0, j) for j = 1, ..., N
    L[0, 0] = 0.0  # Empty sequence has 0 cost
    for j in range(1, N + 1):
        L[0, j] = compute_variance(0, j)
    
    # Fill DP table for i >= 1 change-points
    # Complexity: O(m_max * N^2) as required
    for i in range(1, m_max + 1):
        for j in range(i + 1, N + 1):  # Need at least i+1 elements for i change-points
            # Try all positions t for the i-th change-point
            # t ranges from i to j-1 (need at least i elements before t, and at least 1 after)
            for t in range(i, j):
                cost = L[i - 1, t] + compute_variance(t, j)
                if cost < L[i, j]:
                    L[i, j] = cost
                    backtrack[i, j] = t
    
    # ============================================================
    # STEP 6: Model Selection - Algorithm 1, Line 5
    # Penalty function: g(m, n) = m * (log(n/m) + 1)
    # where m = number of change-points
    # 
    # Select m* = argmin_m [L[m, N] + C * g(m, N)]
    # ============================================================
    if num_segments is None:
        def penalty_function(m, n):
            """
            Penalty function from the paper.
            m: number of change-points
            n: sequence length
            """
            if m <= 0:
                return 0.0  # No penalty for 0 change-points
            if m >= n:
                return np.inf  # Cannot have more change-points than elements
            return m * (np.log(n / m) + 1)
        
        # Find optimal number of change-points
        best_m = 0
        best_score = np.inf
        
        for m in range(0, m_max + 1):
            if L[m, N] == np.inf:
                continue
            score = L[m, N] + penalty_coef * penalty_function(m, N)
            if score < best_score:
                best_score = score
                best_m = m
        
        m_star = best_m
    else:
        # Use user-specified number of segments
        m_star = min(num_segments - 1, m_max)
    
    # ============================================================
    # STEP 7: Backtracking - Algorithm 1, Line 6
    # Find change-point positions t_0, ..., t_{m*-1}
    # These are the internal boundaries (not including 0 and N)
    # ============================================================
    change_points = []
    current_pos = N
    current_m = m_star
    
    # Trace back through the DP table
    while current_m > 0:
        prev_pos = backtrack[current_m, current_pos]
        change_points.append(prev_pos)
        current_pos = prev_pos
        current_m -= 1
    
    # Add boundaries: 0 (start) and N (end)
    # Result: [0, t_1, t_2, ..., t_m, N]
    # Segments are: [0, t_1), [t_1, t_2), ..., [t_m, N)
    change_points = [0] + sorted(change_points) + [N]
    
    return change_points


class KTSSegmentor:
    """Segment videos into events using Kernel Temporal Segmentation."""
    
    def __init__(self, dataset_root, num_segments=None, penalty_coef=1.0):
        """
        Initialize KTS Segmentor.
        
        Args:
            dataset_root: Root directory of dataset (e.g., 'dataset/MSRVTT')
            num_segments: Target number of segments (None for automatic selection)
            penalty_coef: Penalty coefficient for model selection (default: 1.0)
        """
        self.dataset_root = Path(dataset_root)
        self.features_dir = self.dataset_root / 'features'
        self.segments_dir = self.dataset_root / 'segments'
        self.num_segments = num_segments
        self.penalty_coef = penalty_coef
        
        # Create output directory
        self.segments_dir.mkdir(parents=True, exist_ok=True)
    
    def segment_video(self, video_id, skip_existing=True):
        """
        Segment a single video into events.
        
        Args:
            video_id: Video identifier
            skip_existing: Skip if segments already exist
            
        Returns:
            List of segment tuples (start_frame, end_frame)
        """
        # Check if segments already exist
        output_path = self.segments_dir / f"{video_id}.json"
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping {video_id} (segments already exist)")
            with open(output_path, 'r') as f:
                data = json.load(f)
                return data['segments']
        
        # Load features
        features_path = self.features_dir / f"{video_id}.npy"
        if not features_path.exists():
            logger.warning(f"Features not found for {video_id}")
            return None
        
        # Load frame metadata to get original frame indices
        frames_metadata_path = self.dataset_root / 'frames' / video_id / 'frames_metadata.json'
        frame_indices_mapping = None
        if frames_metadata_path.exists():
            try:
                with open(frames_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    frame_indices_mapping = metadata.get('frame_indices', None)
            except Exception as e:
                logger.warning(f"Failed to load frame metadata for {video_id}: {e}")
        
        try:
            features = np.load(features_path)
            
            # Detect change points using KTS algorithm
            change_points = compute_change_points_kts(
                features, 
                num_segments=self.num_segments,
                penalty_coef=self.penalty_coef
            )
            
            # Convert change points to segments
            segments = []
            for i in range(len(change_points) - 1):
                start_idx = int(change_points[i])
                end_idx = int(change_points[i + 1])
                is_last_segment = (i == len(change_points) - 2)
                
                # Map to original frame indices if metadata exists
                if frame_indices_mapping is not None and len(frame_indices_mapping) > 0:
                    # Get original frame index at start position
                    start_frame_orig = frame_indices_mapping[min(start_idx, len(frame_indices_mapping) - 1)]
                    
                    # Get original frame index at end position
                    if is_last_segment:
                        # Last segment: use the very last frame of the video
                        end_frame_orig = frame_indices_mapping[-1]
                    else:
                        # Note: end_idx is exclusive boundary, use it directly for the boundary frame
                        end_frame_orig = frame_indices_mapping[min(end_idx, len(frame_indices_mapping) - 1)]
                    
                    segments.append({
                        'start_frame': int(start_frame_orig),
                        'end_frame': int(end_frame_orig),
                        'start_idx': start_idx,  # Keep extracted frame index for reference
                        'end_idx': end_idx
                    })
                else:
                    # Fallback: use extracted frame indices
                    logger.warning(f"No frame metadata found for {video_id}, using extracted indices")
                    segments.append({
                        'start_frame': start_idx,
                        'end_frame': end_idx
                    })
            
            # Save segments
            output_data = {
                'video_id': video_id,
                'num_frames': int(features.shape[0]),
                'num_segments': len(segments),
                'segments': segments
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.debug(f"Segmented {video_id} into {len(segments)} events")
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting {video_id}: {str(e)}")
            return None
    
    def process_dataset(self, skip_existing=True, whitelist_video_ids=None):
        """
        Process all videos in the dataset.
        
        Args:
            skip_existing: Skip videos that already have segments
            whitelist_video_ids: Set of video IDs to process (None = process all)
            
        Returns:
            Dictionary with processing statistics
        """
        if not self.features_dir.exists():
            logger.error(f"Features directory not found: {self.features_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        # Get all feature files
        feature_files = list(self.features_dir.glob('*.npy'))
        
        if not feature_files:
            logger.warning(f"No feature files found in {self.features_dir}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}
        
        logger.info(f"Found {len(feature_files)} videos to segment")
        
        stats = {'processed': 0, 'skipped': 0, 'failed': 0, 'filtered': 0}
        
        # Process each video
        for feature_file in tqdm(feature_files, desc="Segmenting videos"):
            video_id = feature_file.stem
            
            # Filter by whitelist if provided
            if whitelist_video_ids is not None and video_id not in whitelist_video_ids:
                logger.debug(f"Filtering out {video_id} (not in training split)")
                stats['filtered'] += 1
                continue
            
            # Check if already processed
            if skip_existing and (self.segments_dir / f"{video_id}.json").exists():
                stats['skipped'] += 1
                continue
            
            # Segment video
            segments = self.segment_video(video_id, skip_existing=False)
            
            if segments is not None:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Segmentation complete. Processed: {stats['processed']}, "
                   f"Skipped: {stats['skipped']}, Failed: {stats['failed']}, "
                   f"Filtered: {stats.get('filtered', 0)}")
        
        return stats


def segment_videos(dataset_name, dataset_root='dataset', num_segments=None, penalty_coef=1.0):
    """
    Convenience function to segment videos in a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MSRVTT')
        dataset_root: Root directory containing datasets
        num_segments: Target number of segments per video (None for automatic)
        penalty_coef: Penalty coefficient for model selection (default: 1.0)
        
    Returns:
        Processing statistics
    """
    dataset_path = Path(dataset_root) / dataset_name
    segmentor = KTSSegmentor(dataset_path, num_segments=num_segments, penalty_coef=penalty_coef)
    return segmentor.process_dataset()


if __name__ == "__main__":
    # Test the segmentor
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment videos using KTS (Potapov et al. 2014)')
    parser.add_argument('--dataset_name', type=str, default='MSRVTT',
                       help='Name of the dataset')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Root directory containing datasets')
    parser.add_argument('--num_segments', type=int, default=None,
                       help='Target number of segments (None for automatic selection)')
    parser.add_argument('--penalty_coef', type=float, default=1.0,
                       help='Penalty coefficient for model selection (default: 1.0)')
    
    args = parser.parse_args()
    
    stats = segment_videos(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        num_segments=args.num_segments,
        penalty_coef=args.penalty_coef
    )
    
    print(f"\nSegmentation complete:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
