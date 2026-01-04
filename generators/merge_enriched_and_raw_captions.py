"""
Step 5: Merge enriched captions (train) with raw captions (test + val) for MSVD dataset

This script creates a complete enriched-caption.pkl file that contains:
- Enriched captions for training videos (from enriched-caption.pkl)
- Raw captions for test and validation videos (from raw-captions.pkl)

Output: A new enriched-caption.pkl file with all videos
"""

import os
import pickle
from pathlib import Path


def merge_msvd_captions(dataset_path):
    """
    Merge enriched captions (train) with raw captions (test + val)
    
    Args:
        dataset_path: Path to MSVD dataset folder
    """
    dataset_path = Path(dataset_path)
    
    print("=" * 80)
    print("STEP 5: MERGE ENRICHED CAPTIONS WITH RAW CAPTIONS FOR TEST + VAL")
    print("=" * 80)
    
    # File paths
    enriched_caption_file = dataset_path / 'enriched-captions.pkl'
    raw_caption_file = dataset_path / 'raw-captions.pkl'
    test_list_file = dataset_path / 'test_list.txt'
    val_list_file = dataset_path / 'val_list.txt'
    output_file = dataset_path / 'enriched-caption-complete.pkl'
    backup_file = dataset_path / 'enriched-caption.pkl.backup'
    
    # Check if files exist
    if not enriched_caption_file.exists():
        print(f"❌ Error: {enriched_caption_file} not found!")
        return
    
    if not raw_caption_file.exists():
        print(f"❌ Error: {raw_caption_file} not found!")
        return
    
    # Load enriched captions (train set)
    print(f"\n📂 Loading enriched captions from: {enriched_caption_file.name}")
    with open(enriched_caption_file, 'rb') as f:
        enriched_captions = pickle.load(f)
    print(f"   Loaded {len(enriched_captions)} videos with enriched captions")
    
    # Load raw captions (all videos)
    print(f"\n📂 Loading raw captions from: {raw_caption_file.name}")
    with open(raw_caption_file, 'rb') as f:
        raw_captions = pickle.load(f)
    print(f"   Loaded {len(raw_captions)} videos with raw captions")
    
    # Load test video IDs
    print(f"\n📂 Loading test video list from: {test_list_file.name}")
    with open(test_list_file, 'r') as f:
        test_video_ids = [line.strip() for line in f.readlines()]
    print(f"   Loaded {len(test_video_ids)} test video IDs")
    
    # Load val video IDs
    print(f"\n📂 Loading val video list from: {val_list_file.name}")
    with open(val_list_file, 'r') as f:
        val_video_ids = [line.strip() for line in f.readlines()]
    print(f"   Loaded {len(val_video_ids)} val video IDs")
    
    # Create merged captions dictionary
    print(f"\n🔧 Creating merged captions dictionary...")
    merged_captions = {}
    
    # Step 1: Add all enriched captions (train videos)
    print(f"\n   Step 1: Adding enriched captions for train videos...")
    for video_id, captions in enriched_captions.items():
        merged_captions[video_id] = captions
    print(f"   ✓ Added {len(enriched_captions)} train videos with enriched captions")
    
    # Step 2: Add raw captions for test videos
    print(f"\n   Step 2: Adding raw captions for test videos...")
    test_count = 0
    test_missing = 0
    for video_id in test_video_ids:
        if video_id in raw_captions:
            merged_captions[video_id] = raw_captions[video_id]
            test_count += 1
        else:
            print(f"   ⚠️  Warning: Test video '{video_id}' not found in raw captions")
            test_missing += 1
    print(f"   ✓ Added {test_count} test videos with raw captions")
    if test_missing > 0:
        print(f"   ⚠️  {test_missing} test videos not found in raw captions")
    
    # Step 3: Add raw captions for val videos
    print(f"\n   Step 3: Adding raw captions for val videos...")
    val_count = 0
    val_missing = 0
    for video_id in val_video_ids:
        if video_id in raw_captions:
            merged_captions[video_id] = raw_captions[video_id]
            val_count += 1
        else:
            print(f"   ⚠️  Warning: Val video '{video_id}' not found in raw captions")
            val_missing += 1
    print(f"   ✓ Added {val_count} val videos with raw captions")
    if val_missing > 0:
        print(f"   ⚠️  {val_missing} val videos not found in raw captions")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"{'=' * 80}")
    print(f"Total videos in merged captions: {len(merged_captions)}")
    print(f"  - Train (enriched): {len(enriched_captions)}")
    print(f"  - Test (raw): {test_count}")
    print(f"  - Val (raw): {val_count}")
    
    # Check for overlaps (shouldn't happen)
    train_test_overlap = set(enriched_captions.keys()).intersection(set(test_video_ids))
    train_val_overlap = set(enriched_captions.keys()).intersection(set(val_video_ids))
    
    if train_test_overlap:
        print(f"\n⚠️  WARNING: {len(train_test_overlap)} videos appear in both train and test!")
    if train_val_overlap:
        print(f"\n⚠️  WARNING: {len(train_val_overlap)} videos appear in both train and val!")
    
    # Display sample of merged captions
    print(f"\n{'=' * 80}")
    print(f"SAMPLE DATA:")
    print(f"{'=' * 80}")
    
    # Show 1 example from each split
    if enriched_captions:
        train_sample_id = list(enriched_captions.keys())[0]
        print(f"\nTrain video (enriched): {train_sample_id}")
        print(f"  Number of captions: {len(merged_captions[train_sample_id])}")
        print(f"  First caption: {' '.join(merged_captions[train_sample_id][0][:10])}...")
    
    if test_count > 0 and test_video_ids[0] in merged_captions:
        test_sample_id = test_video_ids[0]
        print(f"\nTest video (raw): {test_sample_id}")
        print(f"  Number of captions: {len(merged_captions[test_sample_id])}")
        print(f"  First caption: {' '.join(merged_captions[test_sample_id][0][:10])}...")
    
    if val_count > 0 and val_video_ids[0] in merged_captions:
        val_sample_id = val_video_ids[0]
        print(f"\nVal video (raw): {val_sample_id}")
        print(f"  Number of captions: {len(merged_captions[val_sample_id])}")
        print(f"  First caption: {' '.join(merged_captions[val_sample_id][0][:10])}...")
    
    # Save merged captions
    print(f"\n{'=' * 80}")
    print(f"SAVING OUTPUT:")
    print(f"{'=' * 80}")
    
    # Save to new file first
    print(f"\n💾 Saving merged captions to: {output_file.name}")
    with open(output_file, 'wb') as f:
        pickle.dump(merged_captions, f)
    print(f"   ✓ Saved successfully!")
    
    # Ask user if they want to replace the original file
    print(f"\n{'=' * 80}")
    print(f"OPTIONS:")
    print(f"{'=' * 80}")
    print(f"\nThe merged captions have been saved to: {output_file.name}")
    print(f"\nIf you want to use this as the main enriched-caption.pkl file:")
    print(f"  1. Backup: {enriched_caption_file.name} → {backup_file.name}")
    print(f"  2. Replace: {output_file.name} → {enriched_caption_file.name}")
    print(f"\nOr manually rename the files as needed.")
    
    # Optionally create a backup and replace (commented out for safety)
    print(f"\n💡 To automatically replace the original file, uncomment the code below.")
    
    """
    # Uncomment these lines to automatically backup and replace:
    
    # Backup original enriched-caption.pkl
    if not backup_file.exists():
        print(f"\n📦 Creating backup: {backup_file.name}")
        import shutil
        shutil.copy2(enriched_caption_file, backup_file)
        print(f"   ✓ Backup created")
    
    # Replace with merged version
    print(f"\n🔄 Replacing {enriched_caption_file.name} with merged version...")
    import shutil
    shutil.copy2(output_file, enriched_caption_file)
    print(f"   ✓ Replaced successfully!")
    """
    
    return merged_captions


def verify_merged_captions(dataset_path):
    """
    Verify the merged captions file
    """
    dataset_path = Path(dataset_path)
    output_file = dataset_path / 'enriched-caption-complete.pkl'
    
    if not output_file.exists():
        print(f"❌ Output file not found: {output_file}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"VERIFICATION:")
    print(f"{'=' * 80}")
    
    with open(output_file, 'rb') as f:
        merged_captions = pickle.load(f)
    
    print(f"\n✓ Successfully loaded {output_file.name}")
    print(f"✓ Total videos: {len(merged_captions)}")
    
    # Load lists to verify
    test_list_file = dataset_path / 'test_list.txt'
    val_list_file = dataset_path / 'val_list.txt'
    train_list_file = dataset_path / 'train_list.txt'
    
    with open(test_list_file, 'r') as f:
        test_ids = set(line.strip() for line in f)
    with open(val_list_file, 'r') as f:
        val_ids = set(line.strip() for line in f)
    with open(train_list_file, 'r') as f:
        train_ids = set(line.strip() for line in f)
    
    # Check coverage
    merged_ids = set(merged_captions.keys())
    
    test_coverage = len(test_ids.intersection(merged_ids))
    val_coverage = len(val_ids.intersection(merged_ids))
    train_coverage = len(train_ids.intersection(merged_ids))
    
    print(f"\nCoverage:")
    print(f"  Train: {train_coverage}/{len(train_ids)} ({train_coverage/len(train_ids)*100:.1f}%)")
    print(f"  Val:   {val_coverage}/{len(val_ids)} ({val_coverage/len(val_ids)*100:.1f}%)")
    print(f"  Test:  {test_coverage}/{len(test_ids)} ({test_coverage/len(test_ids)*100:.1f}%)")
    
    if train_coverage == len(train_ids) and val_coverage == len(val_ids) and test_coverage == len(test_ids):
        print(f"\n✅ ALL VIDEOS COVERED! The merged file is complete.")
    else:
        print(f"\n⚠️  Some videos are missing from the merged file.")


def main():
    """Main function"""
    # Set dataset path
    script_dir = Path(__file__).parent.parent
    dataset_path = script_dir / 'datasets' / 'MSVD'
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Merge captions
    merged_captions = merge_msvd_captions(dataset_path)
    
    if merged_captions:
        # Verify the output
        verify_merged_captions(dataset_path)
    
    print(f"\n{'=' * 80}")
    print(f"DONE!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
