import numpy as np
import joblib
import random

def get_datasets(filepath, *, test_percent=0.1):
    episodes = joblib.load(filepath)
    cut = int(len(episodes) * test_percent)
    assert cut > 1 

    trainset = Dataset(episodes[cut:])
    testset = Dataset(episodes[:cut])

    return trainset, testset

class Dataset(object):
    def __init__(self, episodes):

        self._episodes = episodes
        self._size = len(self._episodes)
                
    @property
    def episodes(self):
        return self._episodes
    
    @property
    def size(self):
        return self._size
    
    def next_episode(self):
        """Return next episode examples from this data set."""
      
        i = random.randint(0, self._size - 1)

        images, labels = self._episodes[i]
        images = np.array(images).astype(float)
        labels = np.array(labels).astype(float)
        return images, labels

    def next_episode_batch(self, batch_size):
        batch_images = []
        batch_labels = []
        for _ in range(batch_size):
            images, labels = self.next_episode()
            batch_images.append(images)
            batch_labels.append(labels)
        return batch_images, batch_labels