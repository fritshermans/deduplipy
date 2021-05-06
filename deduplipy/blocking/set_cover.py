import heapq
from typing import List, Set


def greedy_set_cover(subsets: List, parent_set: Set, recall: float = 1.0) -> List:
    """
    Greedy set cover algorithm, stops when recall threshold is reached

    Args:
        subsets: subsets that should cover the parent_set
        parent_set: parent_set that should be covered by subsets
        recall: minimum recall to reach

    Returns:
        list containing selection of rules that collectively span the parent_set

    """
    # algorithm obtained from stackoverflow post
    # https://stackoverflow.com/questions/21973126/set-cover-or-hitting-set-numpy-least-element-combinations-to-make-up-full-set
    if not isinstance(parent_set, set):
        parent_set = set(parent_set)
    subsets = [set(x) if not isinstance(x, set) else x for x in subsets]
    max = len(parent_set)
    heap = []
    for s in subsets:
        # Python's heapq lets you pop the *smallest* value, so we
        # want to use max-len(s) as a score, not len(s).
        # len(heap) is just proving a unique number to each subset,
        # used to tiebreak equal scores.
        heapq.heappush(heap, [max - len(s), len(heap), s])
    results = []
    result_set = set()
    while result_set < parent_set:
        best = []
        unused = []
        while heap:
            score, count, s = heapq.heappop(heap)
            if not best:
                best = [max - len(s - result_set), count, s]
                continue
            if score >= best[0]:
                # because subset scores only get worse as the result_set
                # gets bigger, we know that the rest of the heap cannot beat
                # the best score. So push the subset back on the heap, and
                # stop this iteration.
                heapq.heappush(heap, [score, count, s])
                break
            score = max - len(s - result_set)
            if score >= best[0]:
                unused.append([score, count, s])
            else:
                unused.append(best)
                best = [score, count, s]
        add_set = best[2]
        results.append(add_set)
        result_set.update(add_set)
        coverage = len(result_set.intersection(parent_set)) / len(parent_set)
        if coverage >= recall:
            print(f'recall threshold reached, recall = {coverage}')
            return results
        # subsets that were not the best get put back on the heap for next time.
        while unused:
            heapq.heappush(heap, unused.pop())
    return results
