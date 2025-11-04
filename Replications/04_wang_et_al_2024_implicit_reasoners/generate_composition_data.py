#!/usr/bin/env python
"""
Generate Knowledge Graph Composition Data for Paper 04
Wang et al. (2024) - Grokked Transformers are Implicit Reasoners

This script creates synthetic knowledge graphs for training compositional reasoning.

Task: Given atomic facts (h, r, t), learn to infer compositions:
      If h --r1--> b and b --r2--> t, then (h, r1, r2) --> t

Example:
  Atomic: Paris --capital_of--> France
  Atomic: France --in_continent--> Europe  
  Inferred: Paris --capital_of,in_continent--> Europe
"""

import json
import numpy as np
import random
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
import argparse


def build_dicts(entities):
    """Build entity/relation index mappings"""
    entity2ind = dict()
    ind2entity = []
    for i in range(len(entities)):
        entity = entities[i]
        if not (entity in ind2entity):
            ind2entity.append(entity)
            entity2ind[entity] = len(ind2entity) - 1
    return ind2entity, entity2ind


def choose(arr, ratio_or_count):
    """Choose subset of array"""
    if type(ratio_or_count) == float:
        num = round(ratio_or_count*len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
         assert False
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in rand_inds]


def split(arr, ratio_or_count):
    """Split array into two random subsets"""
    if type(ratio_or_count) == float:
        num = round(ratio_or_count*len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
         assert False
    train, test = [], []
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    for i in range(len(arr)):
        if i in rand_inds:
            train.append(arr[i])
        else:
            test.append(arr[i])
    return [train, test]


def form_items(c, t):
    """
    Format knowledge graph item
    c: list of components (e.g., [entity, relation] or [entity, rel1, rel2])
    t: target entity
    """
    input_text = "".join(c)
    target_text = input_text + "".join([t, "</a>"])
    item = {
        "input_text": input_text,
        "target_text": target_text
    }
    return item


def build_dataset(num_entities, num_relations, out_degree=20, split_train_inferred=True, seed=42):
    """
    Build knowledge graph composition dataset
    
    Args:
        num_entities: Number of entities in the graph
        num_relations: Number of relation types
        out_degree: Average number of outgoing edges per entity
        split_train_inferred: Whether to split into ID/OOD
        seed: Random seed
        
    Returns:
        entities, relations, atomic_facts, inferred_facts (and OOD splits if requested)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Create entities and relations
    entities = ["<e_{}>".format(i) for i in range(num_entities)]
    ind2entity, entity2ind = build_dicts(entities)

    relations = ["<r_{}>".format(i) for i in range(num_relations)]
    ind2relation, relation2ind = build_dicts(relations)

    atomic_dict = dict()   # maps a head entity to a list of (r, t) pairs
    atomic_facts = []
    atomics = []

    print(f"Generating atomic facts for {num_entities} entities...")
    for i in tqdm(range(num_entities)):
        # For each subject entity, randomly select some outgoing relations
        num_rows = out_degree
        selected_rows = np.random.choice(num_relations, size=num_rows, replace=False).tolist()
        for row_idx in selected_rows:
            col_idx = np.random.randint(num_entities)  # pick random tail entity
            h, r, t = ind2entity[i], ind2relation[row_idx], ind2entity[col_idx]
            atomic_facts.append(form_items([h, r], t))
            atomics.append((h, r, t))
            if h not in atomic_dict:
                atomic_dict[h] = []
            atomic_dict[h].append((r, t))
    
    if not split_train_inferred:
        # Simple mode: all inferred facts in training
        print("Generating inferred (composition) facts...")
        inferred_facts = []
        for ent in tqdm(entities):
            if ent not in atomic_dict:
                continue
            for (r1, b) in atomic_dict[ent]:
                if b not in atomic_dict:
                    continue
                for (r2, t) in atomic_dict[b]:
                    inferred_facts.append(form_items([ent, r1, r2], t))
        return entities, relations, atomic_facts, inferred_facts
    
    # Advanced mode: Split into ID/OOD
    print("Splitting into ID/OOD...")
    OOD_ratio = 0.05
    OOD_facts, ID_facts = split(atomics, round(len(atomics)*OOD_ratio))
    OOD_facts, ID_facts = set(OOD_facts), set(ID_facts)

    id_atomic_facts = [form_items([h, r], t) for (h, r, t) in ID_facts]
    ood_atomic_facts = [form_items([h, r], t) for (h, r, t) in OOD_facts]

    print("Generating inferred facts with ID/OOD split...")
    train_inferred_facts, test_inferred_iid, test_inferred_ood = [], [], []
    for ent in tqdm(entities):
        if ent not in atomic_dict:
            continue
        for (r1, b) in atomic_dict[ent]:
            if b not in atomic_dict:
                continue
            for (r2, t) in atomic_dict[b]:
                # Check if uses OOD facts
                if (ent, r1, b) in OOD_facts or (b, r2, t) in OOD_facts:
                    if (ent, r1, b) in OOD_facts and (b, r2, t) in OOD_facts:
                        test_inferred_ood.append(form_items([ent, r1, r2], t))
                    continue
                # Split ID facts into train/test
                if np.random.uniform() > 0.005:  # 99.5% train, 0.5% test
                    train_inferred_facts.append(form_items([ent, r1, r2], t))
                else:
                    test_inferred_iid.append(form_items([ent, r1, r2], t))

    return entities, relations, id_atomic_facts, ood_atomic_facts, train_inferred_facts, test_inferred_iid, test_inferred_ood


def create_composition_dataset(output_dir, num_entities=2000, num_relations=200, 
                               out_degree=20, phi=18.0, seed=42):
    """
    Create composition dataset and save to JSON files
    
    Args:
        output_dir: Directory to save data files
        num_entities: Number of entities (default 2000 from paper)
        num_relations: Number of relations (default 200 from paper)
        out_degree: Edges per entity (default 20)
        phi: Ratio of inferred facts to atomic facts for training
        seed: Random seed for reproducibility
    """
    print("="*80)
    print(f"GENERATING COMPOSITION DATASET")
    print("="*80)
    print(f"Entities: {num_entities}")
    print(f"Relations: {num_relations}")
    print(f"Out-degree: {out_degree}")
    print(f"Phi (train inferred ratio): {phi}")
    print(f"Seed: {seed}")
    print("="*80)
    
    # Generate dataset
    entities, relations, id_atomic_facts, ood_atomic_facts, train_inferred_facts, test_inferred_iid, test_inferred_ood = \
        build_dataset(num_entities, num_relations, out_degree=out_degree, split_train_inferred=True, seed=seed)
    
    print(f"\nGenerated:")
    print(f"  ID atomic facts: {len(id_atomic_facts):,}")
    print(f"  OOD atomic facts: {len(ood_atomic_facts):,}")
    print(f"  Train inferred facts: {len(train_inferred_facts):,}")
    print(f"  Test inferred (IID): {len(test_inferred_iid):,}")
    print(f"  Test inferred (OOD): {len(test_inferred_ood):,}")
    
    # Downsample train_inferred according to phi
    print(f"\nDownsampling train_inferred by phi={phi}...")
    train_inferred_downsampled = choose(train_inferred_facts, round(phi * len(id_atomic_facts)))
    print(f"  Train inferred after downsampling: {len(train_inferred_downsampled):,}")
    
    # Combine all atomics for training
    all_atomics = id_atomic_facts + ood_atomic_facts
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare train/valid/test splits
    train_data = all_atomics + train_inferred_downsampled
    valid_data = test_inferred_iid  # Use IID test for validation
    
    # Test includes various probes
    test_size = 1000  # Sample size for test set
    test_data = []
    
    # Add sampled atomics with type labels
    for item in choose(id_atomic_facts, min(test_size, len(id_atomic_facts))):
        test_item = deepcopy(item)
        test_item["type"] = "id_atomic"
        test_data.append(test_item)
    
    for item in choose(ood_atomic_facts, min(test_size//2, len(ood_atomic_facts))):
        test_item = deepcopy(item)
        test_item["type"] = "ood_atomic"
        test_data.append(test_item)
    
    # Add inferred facts
    for item in choose(train_inferred_downsampled, min(test_size, len(train_inferred_downsampled))):
        test_item = deepcopy(item)
        test_item['type'] = 'train_inferred'
        test_data.append(test_item)
    
    for item in test_inferred_iid:
        test_item = deepcopy(item)
        test_item['type'] = 'test_inferred_iid'
        test_data.append(test_item)
    
    for item in choose(test_inferred_ood, min(test_size, len(test_inferred_ood))):
        test_item = deepcopy(item)
        test_item["type"] = "test_inferred_ood"
        test_data.append(test_item)
    
    # Save datasets
    print(f"\nSaving datasets to {output_dir}/...")
    with open(output_path / "train.json", "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ train.json: {len(train_data):,} examples")
    
    with open(output_path / "valid.json", "w", encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2)
    print(f"  ✓ valid.json: {len(valid_data):,} examples")
    
    with open(output_path / "test.json", "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    print(f"  ✓ test.json: {len(test_data):,} examples")
    
    # Create vocabulary
    vocab = entities + relations + ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
    with open(output_path / "vocab.json", "w", encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    print(f"  ✓ vocab.json: {len(vocab):,} tokens")
    
    print(f"\n{'='*80}")
    print(f"✅ Dataset generation complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Ready for training with main.py")
    print("="*80)
    
    return train_data, valid_data, test_data, vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate composition dataset for grokking')
    parser.add_argument('--output_dir', type=str, default='data/composition_minimal',
                       help='Output directory for dataset')
    parser.add_argument('--num_entities', type=int, default=500,
                       help='Number of entities (default 500 for quick testing, paper uses 2000)')
    parser.add_argument('--num_relations', type=int, default=50,
                       help='Number of relations (default 50 for quick testing, paper uses 200)')
    parser.add_argument('--out_degree', type=int, default=20,
                       help='Average outgoing edges per entity')
    parser.add_argument('--phi', type=float, default=18.0,
                       help='Ratio of inferred to atomic facts in training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PAPER 04: KNOWLEDGE GRAPH DATA GENERATION")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Entities: {args.num_entities}")
    print(f"  Relations: {args.num_relations}")
    print(f"  Out-degree: {args.out_degree}")
    print(f"  Phi: {args.phi}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output_dir}")
    print("\nNote: Use --num_entities=2000 --num_relations=200 for full paper replication")
    print("      (but this will take much longer to train)")
    print("="*80 + "\n")
    
    create_composition_dataset(
        output_dir=args.output_dir,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        out_degree=args.out_degree,
        phi=args.phi,
        seed=args.seed
    )

