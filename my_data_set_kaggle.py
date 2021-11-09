from PIL import Image
from torch.utils.data import Dataset
from pandas import read_csv


# from torchvision.transforms import Compose, ToTensor


def get_name_dict(path):
    # path = '../data/classify-leaves/train.csv'
    df = read_csv(path)
    name = df['label']
    name = dict([(e, 0) for e in name])
    name = dict([(e, i) for i, e in enumerate(name)])
    return name


class MyDataset(Dataset):
    def __init__(self, root='', csv_path='', transform=None):
        path = f'{root}{csv_path}'
        fh = read_csv(path).values
        name_dic = get_name_dict(path)
        self.images = [(f'{root}{img_path}', name_dic[name]) for img_path, name in fh]
        self.transform = transform
        # print(self.images)

    def __getitem__(self, index):
        fn, label = self.images[index]
        # img = Image.open(fn).convert('RGB')  # 把图像转成RGB
        img = Image.open(fn).convert('L')  # 把图像转成RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, label  # 这就返回一个样本

    def __len__(self):
        return len(self.images)

#
# if __name__ == '__main__':
#     trans = Compose([ToTensor()])
#     dataset = MyDataset(root='../data/classify-leaves/', csv_path='train.csv', transform=trans)
#     print(dataset[0])
