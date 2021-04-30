import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np

'''
自定义的数据加载格式
'''

# 图像标准化
transform_BZ = transforms.Normalize(
    mean = [0.5, 0.5, 0.5],
    std  = [0.5, 0.5, 0.5]
)


# 数据集类
class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transform_BZ,
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transform_BZ,
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            # list返回,map迭代,lambda匿名函数,参数列表
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
            # 结果：[path, classes]
        return imgs_info

    ''' 防止尺寸太小，加黑边 '''
    def padding_black(self, img):
        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w*scale, h*scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        #img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)


''' 主函数 '''
if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("train num:", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset
                                            , batch_size = 32
                                            , shuffle=True
                                            )

    images, targets = next(iter(train_loader))
    print("train:", images.shape)
    print("train:", targets.shape)

    test_dataset = LoadData("test.txt", True)
    print("test num:", len(test_dataset))

    for i in range(3):
        img = images[10+i]      
        img = img.numpy()   # FloatTensor转为ndarray
        img = np.transpose(img, (1,2,0))    
        plt.imshow(img)
        plt.show()
