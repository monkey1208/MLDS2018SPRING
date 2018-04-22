import torch
import torch.utils.data as data
class Dataset(data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images
        '''
        ids = []
        captions = []
        images = []
        for label in self.labels:
            id = label['id']
            caption = self.rng.choice(label['caption'])
            #image = np.load(os.path.join(root, '{}.npy'.format(id)))
            ids.append(id)
            captions.append(caption)
            images.append(image)
        '''

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption = self.labels[index]
        image = self.images[index]

        # Convert caption (string) to word ids.
        '''
        tokens = caption.lower().strip('.').split()
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        '''
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