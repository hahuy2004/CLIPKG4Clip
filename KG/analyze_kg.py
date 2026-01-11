"""
Knowledge Graph Statistics and Visualization
Displays detailed statistics about the generated KG JSON file
"""

import json
import sys
from collections import Counter

def analyze_kg(json_file):
    """Analyze and display statistics for a KG JSON file."""
    
    print("="*80)
    print(f"KNOWLEDGE GRAPH ANALYSIS: {json_file}")
    print("="*80)
    
    # Load KG
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file}' not found!")
        return
    
    # Basic statistics
    total_heads = len(kg_data)
    total_edges = sum(len(node["neighbors"]) for node in kg_data.values())
    
    # POS distribution
    pos_distribution = Counter(node["pos"] for node in kg_data.values())
    
    # Weight statistics
    all_weights = []
    all_counts = []
    strong_edges = 0
    
    for node in kg_data.values():
        for neighbor in node["neighbors"]:
            all_weights.append(neighbor["weight"])
            all_counts.append(neighbor["count"])
            if neighbor["is_strong"]:
                strong_edges += 1
    
    # Relation statistics
    relation_counter = Counter()
    for node in kg_data.values():
        for neighbor in node["neighbors"]:
            relation_counter[neighbor["relation"]] += 1
    
    # Node degree statistics (number of neighbors)
    degrees = [len(node["neighbors"]) for node in kg_data.values()]
    
    # Display statistics
    print(f"\n📊 BASIC STATISTICS")
    print(f"   • Total Nodes (Heads): {total_heads:,}")
    print(f"   • Total Edges: {total_edges:,}")
    print(f"   • Average Degree: {sum(degrees)/len(degrees):.2f}")
    print(f"   • Max Degree: {max(degrees):,}")
    print(f"   • Min Degree: {min(degrees):,}")
    
    print(f"\n📈 EDGE STATISTICS")
    print(f"   • Strong Edges: {strong_edges:,} ({strong_edges/total_edges*100:.2f}%)")
    print(f"   • Weak Edges: {total_edges-strong_edges:,} ({(total_edges-strong_edges)/total_edges*100:.2f}%)")
    print(f"   • Avg Weight: {sum(all_weights)/len(all_weights):.4f}")
    print(f"   • Max Weight: {max(all_weights):.4f}")
    print(f"   • Min Weight: {min(all_weights):.4f}")
    print(f"   • Avg Count: {sum(all_counts)/len(all_counts):.2f}")
    
    print(f"\n🏷️  POS TAG DISTRIBUTION (Heads)")
    for pos, count in pos_distribution.most_common(10):
        percentage = count / total_heads * 100
        bar = "█" * int(percentage / 2)
        print(f"   {pos:10s} {count:6,} ({percentage:5.2f}%) {bar}")
    
    print(f"\n🔗 TOP 20 MOST COMMON RELATIONS")
    for i, (relation, count) in enumerate(relation_counter.most_common(20), 1):
        percentage = count / total_edges * 100
        print(f"   {i:2}. {relation:20s} {count:6,} ({percentage:5.2f}%)")
    
    print(f"\n🌟 TOP 10 NODES BY DEGREE (Most Connected)")
    nodes_by_degree = sorted(
        [(head, len(data["neighbors"]), data["global_count"]) 
         for head, data in kg_data.items()],
        key=lambda x: x[1],
        reverse=True
    )
    for i, (head, degree, global_count) in enumerate(nodes_by_degree[:10], 1):
        print(f"   {i:2}. {head:25s} → {degree:4} neighbors (global_count: {global_count:,})")
    
    print(f"\n💪 TOP 10 STRONGEST EDGES (Highest Weight)")
    all_edges_with_info = []
    for head, data in kg_data.items():
        for neighbor in data["neighbors"]:
            all_edges_with_info.append((
                head,
                neighbor["tail"],
                neighbor["relation"],
                neighbor["weight"],
                neighbor["count"]
            ))
    
    all_edges_with_info.sort(key=lambda x: x[3], reverse=True)
    
    for i, (head, tail, relation, weight, count) in enumerate(all_edges_with_info[:10], 1):
        print(f"   {i:2}. {head:15s} --[{relation}]--> {tail:15s} "
              f"(weight: {weight:.4f}, count: {count})")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def main():
    """Main function."""
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default files
        import os
        
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

        # files = ["msrvtt_weighted_kg.json", "msvd_weighted_kg.json"]
        files = [
            os.path.join(SCRIPT_DIR, "msrvtt_weighted_kg.json"),
            os.path.join(SCRIPT_DIR, "msvd_weighted_kg.json")
        ]
        
        print("Select KG file to analyze:")
        for i, f in enumerate(files, 1):
            exists = "✓" if os.path.exists(f) else "✗"
            print(f"  {i}. {f} {exists}")
        
        choice = input("\nEnter number (1-2) or filename: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            json_file = files[int(choice) - 1]
        else:
            json_file = choice
    
    analyze_kg(json_file)


if __name__ == "__main__":
    main()
