class PrioritizedBuffer:
    def __init__(self, capacity=10000):
        self.SumTree(capacity)


class SumTree:
    def __init__(self, capacity=10000):
        '''
        a 0-index array, the entire size of array is at least 2 * capacity, so that the tree is a complete binary tree
        '''
        
        self.capacity = capacity
        self.size = self._get_size(self.capacity)


    def _get_size(self, capacity):
        self.capacity
        size = 1
        while size < capacity * 2:
            size *= 2

        return size