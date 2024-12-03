from itertools import combinations

class KFoldSplitter:
    def __init__(self, match_ids, val_size):
        """
        Initialize the KFoldSplitter.

        :param match_ids: List of match IDs.
        :param val_size: Number of matches in each validation fold.
        """
        self.match_ids = match_ids
        self.val_size = val_size
        self.splits = []

    def split(self):
        """
        Generate train and validation splits.
        """
        val_combinations = list(combinations(self.match_ids, self.val_size))
        self.splits = [
            {
                "train_indices": sorted(list(set(self.match_ids) - set(val_set))),
                "val_indices": sorted(list(val_set))
            }
            for val_set in val_combinations
        ]

        return self.splits

    def get_splits(self):
        """
        Get the generated splits.

        :return: List of dictionaries with train and validation indices.
        """
        if not self.splits:
            raise ValueError("Splits not generated yet. Call 'generate_splits' first.")
        return self.splits