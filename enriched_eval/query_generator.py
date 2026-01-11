"""
Query Generator using LLM (GPT-4) for Offline Pre-processing
Phase 1: Generate enriched query variations
"""

import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

PROMPT_TEMPLATE = """You are given a caption describing a visual scene. Your task is to rewrite the caption into {n} different sentences following the rules:
1. You can diversify the sentence structure and word usage, but you should strictly keep the same semantic meaning.
2. Do not add uncertain details that do not associate with the visual scene. The rewriting should strictly follow the factual information in the original caption.
3. The rewritten captions should be diverse in number of words.
4. The rewritten captions should be no more than 10 words longer than the original caption.

The input caption is: {caption}

Please output ONLY the {n} rewritten sentences, one per line, without numbering or any additional text."""


def generate_enriched_queries(
    input_captions,
    output_json_path,
    api_key,
    n_variations=10,
    model="gpt-4",
    batch_size=1,
    sleep_time=1.0
):
    """
    Generate enriched query variations using GPT-4.
    
    Args:
        input_captions: List of original captions or dict with video_id -> caption
        output_json_path: Path to save enriched data
        api_key: OpenAI API key
        n_variations: Number of variations to generate (default: 10)
        model: OpenAI model to use (default: gpt-4)
        batch_size: Process captions in batches (default: 1)
        sleep_time: Sleep between API calls to avoid rate limits
        
    Returns:
        dict: {video_id: [original_caption, variation1, ..., variation_n]}
    """
    client = OpenAI(api_key=api_key)
    
    # Convert to dict format if input is list
    if isinstance(input_captions, list):
        input_captions = {f"video_{i}": cap for i, cap in enumerate(input_captions)}
    
    enriched_data = {}
    
    # Check if output file exists for resuming
    if os.path.exists(output_json_path):
        print(f"Loading existing enriched data from {output_json_path}")
        with open(output_json_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
    
    print(f"Generating enriched queries for {len(input_captions)} captions...")
    print(f"Using model: {model}, variations per caption: {n_variations}")
    
    for video_id, original_caption in tqdm(input_captions.items()):
        # Skip if already processed
        if video_id in enriched_data:
            continue
        
        try:
            prompt = PROMPT_TEMPLATE.format(n=n_variations, caption=original_caption)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites captions while preserving semantic meaning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse response
            variations_text = response.choices[0].message.content.strip()
            variations = [line.strip() for line in variations_text.split('\n') if line.strip()]
            
            # Ensure we have exactly n variations
            if len(variations) < n_variations:
                print(f"Warning: Only got {len(variations)} variations for {video_id}, padding with original")
                variations += [original_caption] * (n_variations - len(variations))
            elif len(variations) > n_variations:
                variations = variations[:n_variations]
            
            # Store as [original, var1, var2, ..., var_n]
            enriched_data[video_id] = [original_caption] + variations
            
            # Save periodically
            if len(enriched_data) % 10 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2, ensure_ascii=False)
            
            # Rate limiting
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            # Store original only if error
            enriched_data[video_id] = [original_caption] * (n_variations + 1)
            continue
    
    # Final save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    print(f"Enriched data saved to {output_json_path}")
    print(f"Total entries: {len(enriched_data)}")
    
    return enriched_data


def load_enriched_queries(json_path):
    """
    Load pre-generated enriched queries from JSON file.
    
    Args:
        json_path: Path to enriched queries JSON
        
    Returns:
        dict: {video_id: [original_caption, variation1, ..., variation_n]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded enriched queries for {len(data)} videos from {json_path}")
    return data


if __name__ == "__main__":
    # Example usage for offline preprocessing
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enriched queries using GPT-4")
    parser.add_argument("--input_json", type=str, required=True, help="Input captions JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Output enriched queries JSON file")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--n_variations", type=int, default=10, help="Number of variations per caption")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    # Load input captions
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_captions = json.load(f)
    
    # Generate enriched queries
    generate_enriched_queries(
        input_captions=input_captions,
        output_json_path=args.output_json,
        api_key=args.api_key,
        n_variations=args.n_variations,
        model=args.model
    )
