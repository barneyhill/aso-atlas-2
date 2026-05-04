"""
Fast similarity detection for near-duplicate tables.

Uses length-based prefiltering and real_quick_ratio() to minimize expensive ratio() calls.
Supports parallel processing across multiple cores.
"""
from __future__ import annotations
import re
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_table_number(filename: str) -> str:
    """Extract normalized table number from filename."""
    m = re.search(r'table_(\d+)', filename, re.IGNORECASE)
    if not m:
        return ''
    return m.group(1).lstrip('0') or '0'


def read_file_preview(file_path: str, max_lines: int = 200) -> str:
    """Read first max_lines from file for comparison."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = [fh.readline() for _ in range(max_lines)]
            return ''.join(lines)
    except Exception:
        return ''


def cheap_length_upper_bound(len_a: int, len_b: int) -> float:
    """
    Very cheap upper bound based only on lengths.
    Max possible SequenceMatcher ratio <= min(len_a, len_b) / max(len_a, len_b).
    """
    if len_a == 0 and len_b == 0:
        return 1.0
    if max(len_a, len_b) == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)


def similarity_with_pruning(content1: str, content2: str, len1: int, len2: int,
                            threshold: float) -> float:
    """
    Returns true ratio if >= threshold; otherwise returns 0.0.
    Uses length-based prefilter and real_quick_ratio() to minimize expensive ratio() calls.
    """
    # 1) length-based upper bound (O(1))
    len_ub = cheap_length_upper_bound(len1, len2)
    if len_ub < threshold:
        return 0.0

    # 2) SequenceMatcher.real_quick_ratio() (cheap, safe upper bound)
    sm = SequenceMatcher(None, content1, content2, autojunk=True)
    rq = sm.real_quick_ratio()
    if rq < threshold:
        return 0.0

    # 3) Compute full ratio (expensive)
    r = sm.ratio()
    return r if r >= threshold else 0.0


def process_single_group(args_tuple):
    """Process a single table group - for parallel execution."""
    table_num, group_files, contents, lengths, threshold = args_tuple

    results: List[Tuple[str, str, float]] = []
    n = len(group_files)

    for i in range(n):
        for j in range(i + 1, n):
            f1 = group_files[i]
            f2 = group_files[j]
            sim = similarity_with_pruning(contents[f1], contents[f2],
                                          lengths[f1], lengths[f2],
                                          threshold)
            if sim >= threshold:
                results.append((f1, f2, sim))

    return results


def find_similar_pairs(files: List[str], threshold: float = 0.90,
                       max_lines: int = 200, show_progress: bool = True) -> List[Tuple[str, str, float]]:
    """
    Find all pairs of files with similarity >= threshold.
    Only compares files with matching table numbers.

    Args:
        files: List of file paths
        threshold: Similarity threshold (0.0 to 1.0)
        max_lines: Number of lines to compare (default 200)
        show_progress: Show progress bar

    Returns:
        List of (file1, file2, similarity) tuples
    """
    # Group by table number
    table_groups = defaultdict(list)
    for f in files:
        table_num = extract_table_number(f)
        table_groups[table_num].append(f)

    groups_to_process = {k: v for k, v in table_groups.items() if len(v) >= 2}

    if show_progress:
        print(f"Grouped into {len(table_groups)} table numbers; {len(groups_to_process)} groups have 2+ files")
        print(f"Reading file contents (first {max_lines} lines)...")

    # Read contents once
    contents = {}
    for f in files:
        contents[f] = read_file_preview(f, max_lines)
    lengths = {p: len(contents[p]) for p in files}

    # Prepare tasks for parallel processing
    tasks = [
        (table_num, group_files, contents, lengths, threshold)
        for table_num, group_files in sorted(groups_to_process.items())
    ]

    # Process groups in parallel
    num_workers = os.cpu_count()
    if show_progress:
        print(f"Processing {len(tasks)} groups in parallel with {num_workers} workers...")

    high_similarity: List[Tuple[str, str, float]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_group, task): task for task in tasks}

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Similarity detection", unit="group")

        for future in iterator:
            results = future.result()
            high_similarity.extend(results)

    return high_similarity


def resolve_duplicate_chains(similarity_pairs: List[Tuple[str, str, float]]) -> Dict[str, str]:
    """
    Resolve chains of duplicates to find canonical (first) instance.

    Example: If A~B (90% similar) and B~C (95% similar), then:
    - A is canonical (alphabetically first)
    - B links to A
    - C links to A (transitive)

    Args:
        similarity_pairs: List of (file1, file2, similarity) tuples

    Returns:
        Dict mapping every file to its canonical version
    """
    # Build adjacency graph
    graph = defaultdict(set)
    all_files = set()

    for f1, f2, sim in similarity_pairs:
        graph[f1].add(f2)
        graph[f2].add(f1)
        all_files.add(f1)
        all_files.add(f2)

    # Find connected components using DFS
    visited = set()
    file_to_canonical = {}

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for file in all_files:
        if file not in visited:
            component = []
            dfs(file, component)

            # Choose canonical as alphabetically first (deterministic)
            canonical = min(component)

            # Map all files in component to canonical
            for f in component:
                file_to_canonical[f] = canonical

    return file_to_canonical
