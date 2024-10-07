from collections import defaultdict

import numpy as np


class FeatureExtraction:
  def __init__(self) -> None:
    pass
  
  def create_frequency_matrix(self,corpus,vocab_list=None) -> tuple:
        frequency_matrix = []
        
        if vocab_list is None:
            vocabulary = defaultdict(int)
            for text in corpus:
                for word in text:
                    vocabulary[word] += 1
            vocab_list = list(vocabulary.keys())
        
        for text in corpus:
            word_count = [text.count(word) for word in vocab_list]
            frequency_matrix.append(word_count)
        
        return np.array(frequency_matrix), vocab_list
