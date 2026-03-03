from itertools import combinations
from typing import Callable, List


def greedy_group_search(
    num_items: int,
    group_size: int,
    score_fn: Callable[[List[int], int], float],
) -> List[List[int]]:
    """
    Greedily partitions num_items into groups of group_size by maximizing score_fn.
    
    Args:
        num_items: The total number of items to group.
        group_size: The number of items in each group.
        score_fn: A function that takes a list of candidate indices and the 
                  current slot index (e.g. 0 for the first group), and returns
                  a float score representing the quality of that candidate.
                  
    Returns:
        A list of grouped indices.
    """
    remaining = list(range(num_items))
    groups_idx: List[List[int]] = []
    num_groups = num_items // group_size
    
    for slot in range(num_groups - 1):
        best_score = float("-inf")
        best_combo = None
        for combo in combinations(remaining, group_size):
            score = score_fn(list(combo), slot)
            if score > best_score:
                best_score = score
                best_combo = list(combo)
        
        # In case something fails, fallback to the first combo
        if best_combo is None:
            best_combo = list(remaining[:group_size])
            
        groups_idx.append(best_combo)
        remaining = [i for i in remaining if i not in best_combo]
        
    groups_idx.append(remaining)
    return groups_idx
