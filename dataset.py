from torch.utils.data import Dataset
from PIL import Image
import os
root_path = "./dataset/train"
label_path = "ants"

#打开文件
img = Image.open('./dataset/train/ants/0013035.jpg')
img.show()

#采用DataSet读取
class MyDataSet(Dataset):
    #需要重写 __init__, __getitem__, __len__
    def __init__(self, root_dir , label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir , self.label_dir)
        self.img_path = os.listdir(self.path)
    
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img ,label
    
    #返回总数量
    def __len__(self):
        return len(self.img_path) 

#产生dataset
ants_dataset = MyDataSet(root_dir='.\\dataset\\train' , label_dir="ants")

    