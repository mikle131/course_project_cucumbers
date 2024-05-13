import random
import os
from torch.utils.data import Dataset
from IPython.display import clear_output
from PIL import Image


def shuffle_lists(list1, list2, seed=None):
    if seed is not None:
        random.seed(seed)
    combined_lists = list(zip(list1, list2))
    random.shuffle(combined_lists)
    shuffled_list1, shuffled_list2 = zip(*combined_lists)
    shuffled_list1 = list(shuffled_list1)
    shuffled_list2 = list(shuffled_list2)
    return shuffled_list1, shuffled_list2


class Data(Dataset):
    def __init__(self, data_folder, train = True, train_size = 0.9, transform=None, seed=42):
        super().__init__()
        self.train = train
        self.transform = transform
        self.decode = ['Downy Mildew', 'Bacterial Wilt', 'Fresh Leaf', 'Anthracnose', 'Gummy Stem Blight']
        self.paths = [data_folder + '/' * (data_folder[-1] != '/') + i for i in self.decode]
        images, classes = [], []
        for number_class, path in enumerate(self.paths):
            data = [path + '/' * (path[-1] != '/') + i for i in os.listdir(path)]
            images += data
            classes += [number_class] * len(data)

        images, classes = shuffle_lists(images, classes, seed=seed)
        n = int(len(images) * train_size)
        if train:
            self.classes = classes[:n]
            self.images = self._load_images(images[:n])
        else:
            self.classes = classes[n+1:]
            self.images = self._load_images(images[n+1:])

    def decode_class(self, _class):
        if 0 <= _class <= len(self.decode):
            return self.decode[_class]
        return -1
    def _load_images(self, image_paths):
        images = []
        cnt = 0
        for filename in image_paths:
            if cnt % 10 == 0:
                clear_output()
                print(f'{cnt}/{len(image_paths)}')
            cnt += 1
            image = Image.open(filename).convert('RGB')

            images += [image]
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        label = self.classes[item]
        image = self.images[item]
        image = self.transform(image)
        return image, label