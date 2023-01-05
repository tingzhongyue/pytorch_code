from PIL import Image
from torchvision import transforms

#获取图片
img_path = r".\\dataset\train\ants\0013035.jpg"
img = Image.open(img_path)

#转化为tensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

#常见transforms

#进行归一化
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
tensor_img_norm = trans_norm(tensor_img)
print(tensor_img[0][0][0])
print(tensor_img_norm[0][0][0])

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
print(img_resize.size())

#Compose
trans_resize2 = transforms.Resize((512,512))
trans_compose = transforms.Compose([tensor_trans , trans_resize2])
img_resize_2 = trans_compose(img)
print(img_resize_2.size())

