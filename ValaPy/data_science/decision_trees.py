from typing import List, Any, Dict, TypeVar, NamedTuple, Union
from collections import Counter, defaultdict
from math import log
T = TypeVar('T')

class Decision_Tree():
    def __init__(self):
        pass
    
    def entropy(self, class_probabilities: List[float])->float:
        return sum(-p*log(p,2) for p in class_probabilities if p>0)
    
    def class_probabilities(self, labels: List[Any])->List[float]:
        total_count = len(labels)
        return [count/total_count for count in Counter(labels).values()]
    
    def data_entropy(self, labels: List[Any])->float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: List[List[Any]])->float:
        total_count = sum( len(subset) for subset in subsets )
        return sum(self.data_entropy(subset) * len(subset)/total_count 
                   for subset in subsets)
    
    def partition_by(self, inputs: List[T], attribute: str)-> Dict[Any, List[T]]:
        partitions: Dict[Any, List[T]] = defaultdict(list)
        for input in inputs:
            key = getattr(input, attribute)
            partitions[key].append(input)
        return partitions
    
    def partition_entropy_by(self, inputs: List[Any], attribute: str, label_attribute: str)->float:
        partitions=self.partition_by(inputs, attribute)
        labels = [[getattr(input, label_attribute) for input in partition]
                  for partition in partitions.values()]
        
        return self.partition_entropy(labels)
    
    class Leaf(NamedTuple):
        value: any
        
    class Split(NamedTuple):
        attribute: str
        subtrees: dict
        default_value: Any=None
        
    DecisionTree = Union [Leaf, Split]
    
    def classify(self, tree: DecisionTree, input: Any)->Any:
        if isinstance(tree, self.Leaf):
            return tree.value
        
        subtree_key = getattr(input, tree.attribute)
        
        if subtree_key not in tree.subtrees:
            return tree.default_value
        
        subtree = tree.subtrees[subtree_key]
        return self.classify(subtree, input)
    
    def build_tree_id3(self, inputs: List[Any], split_attributes: List[str], target_attribute: str)-> DecisionTree:
        label_counts = Counter(getattr(input, target_attribute) for input in inputs)
        most_common_label=label_counts.most_common(1)[0][0]
        if len(label_counts)==1:
            return self.Leaf(most_common_label)
        if not split_attributes:
            return self.Leaf(most_common_label)
        def split_entropy(attribute:str)->float:
            return self.partition_entropy_by(inputs, attribute, target_attribute)
        best_attribute = min(split_attributes, key=split_entropy)
        
        partitions = self.partition_by(inputs, best_attribute)
        new_attributes = [a for a in split_attributes if a!=best_attribute]
        
        subtrees = {attribute_value: self.build_tree_id3(subset, new_attributes, target_attribute) 
                    for attribute_value, subset in partitions.items()}
        return self.Split(best_attribute, subtrees, default_value=most_common_label)