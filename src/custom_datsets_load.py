import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = '../data/materials/'
ROOT_PATH_image="/home/mayank_sati/Documents/datsets/"

class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')

            path = osp.join(ROOT_PATH_image, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class datasets_custom(Dataset):

    def __init__(self, setname, imageFolderDataset):
        self.imageFolderDataset = imageFolderDataset
        # csv_path = osp.join(ROOT_PATH, setname + '.csv')
        # lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        # self.wnids = []
        #
        # for l in lines:
        #     name, wnid = l.split(',')
        #
        #     path = osp.join(ROOT_PATH_image, 'images', name)
        #     if wnid not in self.wnids:
        #         self.wnids.append(wnid)
        #         lb += 1
        #     data.append(path)
        #     label.append(lb)
        for values in self.imageFolderDataset.imgs:
            # print(values)
            path=values[0]
            lb=values[1]
            data.append(path)
            label.append(lb)


        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

#
# class SiameseNetworkDataset(Dataset):
#
#     def __init__(self, imageFolderDataset, transform=None, should_invert=True):
#         self.imageFolderDataset = imageFolderDataset
#         self.transform = transform
#         self.should_invert = should_invert
#
#     def __getitem__(self, index):
#         img0_tuple = random.choice(self.imageFolderDataset.imgs)
#         # we need to make sure approx 50% of images are in the same class
#         should_get_same_class = random.randint(0, 1)
#         if should_get_same_class:
#             while True:
#                 # keep looping till the same class image is found
#                 img1_tuple = random.choice(self.imageFolderDataset.imgs)
#                 if img0_tuple[1] == img1_tuple[1]:
#                     break
#         else:
#             while True:
#                 # keep looping till a different class image is found
#
#                 img1_tuple = random.choice(self.imageFolderDataset.imgs)
#                 if img0_tuple[1] != img1_tuple[1]:
#                     break
#
#         img0 = Image.open(img0_tuple[0])
#         img1 = Image.open(img1_tuple[0])
#         ##############################3333
#         # imgk=Image.open(io.BytesIO(img0_tuple[0]))
#         # img0 = list(img0.getdata(0))
#         # img1 = list(img1.getdata(0))
#         img0 = img0.convert("RGB")
#         img1 = img1.convert("RGB")
#         ######################################3
#         # img0 = img0.convert("L")
#         # img1 = img1.convert("L")
#         # list(cool.getdata(0))
#         if self.should_invert:
#             img0 = PIL.ImageOps.invert(img0)
#             img1 = PIL.ImageOps.invert(img1)
#
#         if self.transform is not None:
#             img0 = self.transform(img0)
#             img1 = self.transform(img1)
#             #
#             # img0 = list(img0.getdata(0))
#             # img1 = list(img1.getdata(0))
#
#         return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
#
#     def __len__(self):
#         return len(self.imageFolderDataset.imgs)
#
