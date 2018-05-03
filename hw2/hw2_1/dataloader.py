import torch
import torch.utils.data as data
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption = self.labels[index]
        image = self.images[index]
        # Convert caption (string) to word ids.
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.labels)

class DatasetMultilabel(data.Dataset):
    def __init__(self, images, labels):
        lab = []
        self.label2image = []
        for video in range(len(labels)):
            for label in labels[video]:
                lab.append(label)
                self.label2image.append(video)
        self.labels = np.array(lab)
        #self.labels = labels
        self.images = images

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption = self.labels[index]
        image = self.images[self.label2image[index]]
        # Convert caption (string) to word ids.
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.labels)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (80, 4096).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 80, 4096).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 2D tensor to 3D tensor).
    images = torch.Tensor(images)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap)-1 for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    inputs = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[1:end+1]
        inputs[i, :end] = cap[:end]
    return images, inputs, targets, lengths
