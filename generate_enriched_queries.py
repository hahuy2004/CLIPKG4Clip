"""
Standalone script for generating enriched queries using GPT-4
Run this script BEFORE evaluation to prepare enriched data

Usage for MSRVTT:
    python generate_enriched_queries.py \
        --datatype msrvtt \
        --data_path data/MSRVTT/MSRVTT_JSFUSION_test.csv \
        --output_csv data/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv \
        --output_reference data/MSRVTT/MSRVTT_eval_enriched_reference_data.json \
        --api_key "your-openai-api-key" \
        --n_variations 10

Usage for MSVD:
    python generate_enriched_queries.py \
        --datatype msvd \
        --data_path data/MSVD/test_list.txt \
        --raw_captions data/MSVD/raw-captions.pkl \
        --output_pkl data/MSVD/eval_enriched-caption-complete.pkl \
        --output_reference data/MSVD/enriched_eval_captions.json \
        --api_key "your-openai-api-key" \
        --n_variations 10
"""

import sys
import os
import pickle

# Add parent directory to path to import enriched_eval module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enriched_eval.query_generator import generate_enriched_queries
import argparse
import json
import pandas as pd

# Default API key (can be overridden via command line)
DEFAULT_API_KEY = "YOUR_OPENAI_API_KEY"

def load_msrvtt_csv(csv_path):
    """
    Load MSRVTT test data from CSV file.
    Expected format: query_id, video_id, video_name, caption
    
    Returns:
        dict: {query_id: {'video_id': ..., 'video_name': ..., 'caption': ...}}
    """
    print(f"Loading MSRVTT CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    data = {}
    for _, row in df.iterrows():
        query_id = row['query_id']
        data[query_id] = {
            'video_id': row['video_id'],
            'video_name': row['video_name'],
            'caption': row['caption']
        }
    
    print(f"Loaded {len(data)} MSRVTT queries")
    return data


def load_msvd_data(test_list_path, raw_captions_path):
    """
    Load MSVD test data from test_list.txt and raw-captions.pkl
    
    Args:
        test_list_path: Path to test_list.txt (contains video IDs)
        raw_captions_path: Path to raw-captions.pkl
        
    Returns:
        dict: {video_id: [['token1', 'token2', ...], ...]}
    """
    print(f"Loading MSVD test list from: {test_list_path}")
    with open(test_list_path, 'r') as f:
        test_videos = [line.strip() for line in f if line.strip()]
    
    print(f"Loading MSVD captions from: {raw_captions_path}")
    with open(raw_captions_path, 'rb') as f:
        all_captions = pickle.load(f)
    
    # Filter only test videos
    test_data = {}
    for video_id in test_videos:
        if video_id in all_captions:
            # Take first caption as the main one
            test_data[video_id] = all_captions[video_id]
    
    print(f"Loaded {len(test_data)} MSVD test videos")
    return test_data


def save_msrvtt_enriched_csv(enriched_data, original_data, output_csv_path):
    """
    Save enriched MSRVTT data to CSV.
    Format: query_id, video_id, video_name, caption
    
    For each original query ret{i}:
    - ret{i}: original caption
    - ret{i}_1: enriched variation 1
    - ret{i}_2: enriched variation 2
    - ...
    - ret{i}_10: enriched variation 10
    """
    print(f"Saving enriched MSRVTT CSV to: {output_csv_path}")
    
    rows = []
    for query_id in sorted(original_data.keys()):
        video_id = original_data[query_id]['video_id']
        video_name = original_data[query_id]['video_name']
        
        # Get enriched captions (11 total: 1 original + 10 variations)
        if video_id in enriched_data:
            captions = enriched_data[video_id]
        else:
            # Fallback if enrichment failed
            captions = [original_data[query_id]['caption']] * 11
        
        # Original query
        rows.append({
            'query_id': query_id,
            'video_id': video_id,
            'video_name': video_name,
            'caption': captions[0]
        })
        
        # Enriched queries
        for j in range(1, 11):
            enriched_query_id = f"{query_id}_{j}"
            rows.append({
                'query_id': enriched_query_id,
                'video_id': video_id,
                'video_name': video_name,
                'caption': captions[j] if j < len(captions) else captions[0]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(rows)} rows to {output_csv_path}")


def save_msvd_enriched_pkl(enriched_data, output_pkl_path):
    """
    Save enriched MSVD data to pickle file.
    Format: {video_id: [['token1', 'token2', ...], ...]}
    
    Each video has 11 captions (1 original + 10 enriched)
    """
    print(f"Saving enriched MSVD pickle to: {output_pkl_path}")
    
    output_data = {}
    for video_id, enriched_captions in enriched_data.items():
        # Convert each caption string to list of tokens
        tokenized_captions = []
        for caption in enriched_captions:
            tokens = caption.lower().split()
            tokenized_captions.append(tokens)
        
        output_data[video_id] = tokenized_captions
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Saved {len(output_data)} videos to {output_pkl_path}")


def save_reference_json(enriched_data, output_json_path):
    """
    Save reference JSON for debugging.
    Format: {video_id: [caption1, caption2, ...]}
    """
    print(f"Saving reference JSON to: {output_json_path}")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved reference data to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enriched queries using GPT-4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MSRVTT:
  python generate_enriched_queries.py --datatype msrvtt \\
    --data_path data/MSRVTT/MSRVTT_JSFUSION_test.csv \\
    --output_csv data/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv \\
    --output_reference data/MSRVTT/MSRVTT_eval_enriched_reference_data.json

  # MSVD:
  python generate_enriched_queries.py --datatype msvd \\
    --data_path data/MSVD/test_list.txt \\
    --raw_captions data/MSVD/raw-captions.pkl \\
    --output_pkl data/MSVD/eval_enriched-caption-complete.pkl \\
    --output_reference data/MSVD/enriched_eval_captions.json
        """
    )
    
    # Dataset type
    parser.add_argument("--datatype", type=str, required=True,
                       choices=["msrvtt", "msvd"],
                       help="Dataset type: msrvtt or msvd")
    
    # Input files
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to input data file (CSV for MSRVTT, test_list.txt for MSVD)")
    parser.add_argument("--raw_captions", type=str, default=None,
                       help="Path to raw-captions.pkl (MSVD only)")
    
    # Output files
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV path (MSRVTT only)")
    parser.add_argument("--output_pkl", type=str, default=None,
                       help="Output pickle path (MSVD only)")
    parser.add_argument("--output_reference", type=str, default=None,
                       help="Output reference JSON path (for debugging)")
    
    # GPT-4 parameters
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY,
                       help="OpenAI API key")
    parser.add_argument("--n_variations", type=int, default=10,
                       help="Number of variations per caption (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-4",
                       choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                       help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--sleep_time", type=float, default=1.0,
                       help="Sleep time between API calls in seconds (default: 1.0)")
    
    # Optional
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.datatype == "msrvtt":
        if args.output_csv is None:
            parser.error("--output_csv is required for MSRVTT")
        if args.output_reference is None:
            args.output_reference = args.output_csv.replace('.csv', '_reference.json')
    elif args.datatype == "msvd":
        if args.raw_captions is None:
            parser.error("--raw_captions is required for MSVD")
        if args.output_pkl is None:
            parser.error("--output_pkl is required for MSVD")
        if args.output_reference is None:
            args.output_reference = args.output_pkl.replace('.pkl', '_reference.json')
    
    print("="*70)
    print("ENRICHED QUERY GENERATION")
    print("="*70)
    print(f"Dataset: {args.datatype.upper()}")
    print(f"Model: {args.model}")
    print(f"Variations per caption: {args.n_variations}")
    print("="*70)
    
    # Load data based on dataset type
    if args.datatype == "msrvtt":
        print("\n[MSRVTT] Loading data...")
        original_data = load_msrvtt_csv(args.data_path)
        
        # Extract captions for enrichment
        input_captions = {}
        for query_id, data in original_data.items():
            video_id = data['video_id']
            caption = data['caption']
            # Use video_id as key for enrichment
            if video_id not in input_captions:
                input_captions[video_id] = caption
        
        print(f"Extracted {len(input_captions)} unique videos")
        
    elif args.datatype == "msvd":
        print("\n[MSVD] Loading data...")
        test_data = load_msvd_data(args.data_path, args.raw_captions)
        
        # Extract first caption for each video
        input_captions = {}
        for video_id, captions_list in test_data.items():
            if captions_list and len(captions_list) > 0:
                # Join tokens to form caption
                first_caption = ' '.join(captions_list[0])
                input_captions[video_id] = first_caption
        
        print(f"Extracted {len(input_captions)} videos")
    
    # Apply max_samples limit if specified
    if args.max_samples:
        print(f"\n⚠️  Limiting to first {args.max_samples} samples for testing")
        input_captions = dict(list(input_captions.items())[:args.max_samples])
    
    # Show examples
    print("\n📝 Sample captions:")
    for i, (vid, cap) in enumerate(list(input_captions.items())[:3]):
        print(f"  {vid}: {cap[:80]}...")
    
    # Estimate cost and time
    total_samples = len(input_captions)
    estimated_time_min = (total_samples * args.sleep_time) / 60
    estimated_cost = total_samples * 0.02  # Rough estimate
    
    print(f"\n⏱️  Estimated time: ~{estimated_time_min:.1f} minutes")
    print(f"💰 Estimated cost: ${estimated_cost:.2f} (approximate)")
    print(f"📊 Total API calls: {total_samples}")
    
    # Confirm
    response = input("\n▶️  Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled.")
        return
    
    # Generate enriched queries
    print("\n" + "="*70)
    print("GENERATING ENRICHED QUERIES...")
    print("="*70)
    
    enriched_data = generate_enriched_queries(
        input_captions=input_captions,
        output_json_path=args.output_reference,  # Will save to reference file
        api_key=args.api_key,
        n_variations=args.n_variations,
        model=args.model,
        sleep_time=args.sleep_time
    )
    
    print("\n✅ Query generation completed!")
    print(f"Generated {len(enriched_data)} enriched video captions")
    
    # Save in format-specific output
    print("\n" + "="*70)
    print("SAVING OUTPUT FILES...")
    print("="*70)
    
    if args.datatype == "msrvtt":
        save_msrvtt_enriched_csv(enriched_data, original_data, args.output_csv)
        save_reference_json(enriched_data, args.output_reference)
        
        print(f"\n✅ MSRVTT outputs saved:")
        print(f"  📄 CSV: {args.output_csv}")
        print(f"  📄 Reference JSON: {args.output_reference}")
        
    elif args.datatype == "msvd":
        save_msvd_enriched_pkl(enriched_data, args.output_pkl)
        save_reference_json(enriched_data, args.output_reference)
        
        print(f"\n✅ MSVD outputs saved:")
        print(f"  📄 Pickle: {args.output_pkl}")
        print(f"  📄 Reference JSON: {args.output_reference}")
    
    # Show sample output
    print("\n" + "="*70)
    print("SAMPLE OUTPUT:")
    print("="*70)
    sample_vid = list(enriched_data.keys())[0]
    sample_captions = enriched_data[sample_vid]
    
    print(f"\n{sample_vid}:")
    print(f"  [0] Original: {sample_captions[0]}")
    for j in range(1, min(4, len(sample_captions))):
        print(f"  [{j}] Enriched: {sample_captions[j]}")
    print(f"  ... ({len(sample_captions)} total captions)")
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    
    if args.datatype == "msrvtt":
        print("1. Verify the CSV file has correct format")
        print("2. Use the CSV file for MSRVTT enriched evaluation")
        print(f"   File: {args.output_csv}")
        
    elif args.datatype == "msvd":
        print("1. Verify the pickle file has correct format")
        print("2. Use the pickle file for MSVD enriched evaluation")
        print(f"   File: {args.output_pkl}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
