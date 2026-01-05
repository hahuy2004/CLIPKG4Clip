"""
Merge Split Caption Files
Merges MSRVTT_enriched_data_split_{1,2,3}.json into a single MSRVTT_enriched_data.json file.
"""
import json
from pathlib import Path
import argparse


def merge_split_files(dataset_root, split_files=None, output_file=None):
    """
    Merge split caption files into a single file.
    
    Args:
        dataset_root (str): Path to dataset root directory (e.g., 'datasets/MSRVTT')
        split_files (list): List of split file names. If None, uses default [1,2,3]
        output_file (str): Output filename. If None, uses 'MSRVTT_enriched_data.json'
    
    Returns:
        dict: Merged data with 'videos' and 'sentences' fields
    """
    dataset_path = Path(dataset_root)
    
    # Default split files
    if split_files is None:
        split_files = [
            'MSRVTT_enriched_data_split_1.json',
            'MSRVTT_enriched_data_split_2.json',
            'MSRVTT_enriched_data_split_3.json'
        ]
    
    # Default output file
    if output_file is None:
        output_file = 'MSRVTT_enriched_data.json'
    
    print("=" * 60)
    print("Merging Split Caption Files")
    print("=" * 60)
    
    # Step 1: Load videos metadata from MSRVTT_data.json
    print("\n[1/4] Loading videos metadata from MSRVTT_data.json...")
    msrvtt_data_path = dataset_path / 'MSRVTT_data.json'
    
    videos_data = []
    if msrvtt_data_path.exists():
        try:
            with open(msrvtt_data_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
                videos_data = original_data.get('videos', [])
            print(f"✓ Loaded {len(videos_data)} videos metadata")
        except Exception as e:
            print(f"✗ Error loading MSRVTT_data.json: {e}")
            print("  Using empty 'videos' array")
    else:
        print(f"✗ MSRVTT_data.json not found at {msrvtt_data_path}")
        print("  Using empty 'videos' array")
    
    # Step 2: Load sentences from split files
    print(f"\n[2/4] Loading sentences from {len(split_files)} split files...")
    all_sentences = []
    stats = {'total_files': len(split_files), 'loaded': 0, 'failed': 0}
    
    for split_file in split_files:
        split_path = dataset_path / split_file
        
        if not split_path.exists():
            print(f"  ✗ File not found: {split_file}")
            stats['failed'] += 1
            continue
        
        try:
            with open(split_path, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
                sentences = split_data.get('sentences', [])
                all_sentences.extend(sentences)
                stats['loaded'] += 1
                print(f"  ✓ {split_file}: {len(sentences)} sentences")
        except Exception as e:
            print(f"  ✗ Error loading {split_file}: {e}")
            stats['failed'] += 1
    
    print(f"\nLoaded {stats['loaded']}/{stats['total_files']} files successfully")
    print(f"Total sentences: {len(all_sentences)}")
    
    # Step 3: Create merged data
    print("\n[3/4] Creating merged data structure...")
    merged_data = {
        "videos": videos_data,
        "sentences": all_sentences
    }
    
    # Step 4: Save merged file
    print(f"\n[4/4] Saving merged file to {output_file}...")
    output_path = dataset_path / output_file
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to {output_path}")
        
        # Print file size
        file_size = output_path.stat().st_size
        print(f"  File size: {file_size:,} bytes ({file_size / (1024**2):.2f} MB)")
        
    except Exception as e:
        print(f"✗ Error saving merged file: {e}")
        return None
    
    # Summary
    print("\n" + "=" * 60)
    print("Merge Summary:")
    print(f"  Videos: {len(videos_data)}")
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Files merged: {stats['loaded']}/{stats['total_files']}")
    print(f"  Output: {output_path}")
    print("=" * 60)
    
    return merged_data


def verify_merged_file(merged_file_path):
    """
    Verify the merged file and print statistics.
    
    Args:
        merged_file_path (str): Path to merged file
    """
    file_path = Path(merged_file_path)
    
    if not file_path.exists():
        print(f"✗ File not found: {merged_file_path}")
        return
    
    print("\n" + "=" * 60)
    print("Verifying Merged File")
    print("=" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        videos = data.get('videos', [])
        sentences = data.get('sentences', [])
        
        print(f"\nFile: {file_path.name}")
        print(f"  Videos: {len(videos)}")
        print(f"  Sentences: {len(sentences)}")
        
        # Count sentences per video
        if sentences:
            video_caption_count = {}
            for sent in sentences:
                video_id = sent.get('video_id')
                if video_id:
                    video_caption_count[video_id] = video_caption_count.get(video_id, 0) + 1
            
            print(f"  Unique videos with captions: {len(video_caption_count)}")
            
            if video_caption_count:
                avg_captions = sum(video_caption_count.values()) / len(video_caption_count)
                max_captions = max(video_caption_count.values())
                min_captions = min(video_caption_count.values())
                print(f"  Captions per video: avg={avg_captions:.1f}, min={min_captions}, max={max_captions}")
        
        # Sample sentences
        print("\nSample sentences:")
        for i, sent in enumerate(sentences[:3]):
            video_id = sent.get('video_id', 'N/A')
            caption = sent.get('caption', 'N/A')
            print(f"  [{i+1}] {video_id}: {caption[:80]}...")
        
        print("\n✓ Verification complete")
        
    except Exception as e:
        print(f"✗ Error verifying file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge split caption files into a single file')
    parser.add_argument('--dataset_root', type=str, 
                       default=r'e:\Code_KLTN_E\0_CLIPKG4Clip\datasets\MSRVTT',
                       help='Path to dataset root directory')
    parser.add_argument('--split_files', type=str, nargs='+', default=None,
                       help='List of split file names (default: auto-detect split_{1,2,3})')
    parser.add_argument('--output', type=str, default='MSRVTT_enriched_data.json',
                       help='Output filename')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the merged file after creation')
    
    args = parser.parse_args()
    
    # Merge files
    merged_data = merge_split_files(
        dataset_root=args.dataset_root,
        split_files=args.split_files,
        output_file=args.output
    )
    
    # Verify if requested
    if args.verify and merged_data is not None:
        output_path = Path(args.dataset_root) / args.output
        verify_merged_file(output_path)
