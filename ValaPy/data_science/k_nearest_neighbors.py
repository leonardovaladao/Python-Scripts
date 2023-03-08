from typing import List, NamedTuple
from collections import Counter
from .linear_algebra import Vector, distance

class KNN():
    def __init__(self):
        pass
    
    def raw_majority_vote(self, labels: List[str])->str:
        votes = Counter(labels)
        winner, _ = votes.most_common(1)[0]
        return winner
    
    def majority_vote(self, labels: List[str])->str:
        vote_counts=Counter(labels)
        winner, winner_count = vote_counts.most_common(1)[0]
        num_winners = len([count for count in vote_counts.values() if count==winner_count])
        if num_winners==1:
            return winner
        else:
            return self.majority_vote(labels[:-1])
        
    class LabeledPoint(NamedTuple):
        point: Vector
        label: str
        
    def knn_classify(self, k: int, labeled_points: List[LabeledPoint], new_point: Vector)->str:
        by_distance = sorted(labeled_points, key=lambda lp:distance(lp.point, new_point))
        k_nearest_labels = [lp.label for lp in by_distance[:k]]
        return self.majority_vote(k_nearest_labels)
    