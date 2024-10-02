import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

class KAISTDataset(Dataset):
    def __init__(self, annotations_dir, images_dir, RGBtransform=None, Ttransform=None):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.RGBtransform = RGBtransform
        self.Ttransform = Ttransform
        
        # 获取所有标注文件
        self.annotation_files = os.listdir(annotations_dir)
        
    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        # 解析XML标注文件
        annotation_file = os.path.join(self.annotations_dir, self.annotation_files[idx])
        annotation_data = self.parse_xml_annotation(annotation_file)
        
        # 构建图像路径
        visible_image_path = os.path.join(self.images_dir, 'visible', annotation_data['filename'])
        lwir_image_path = os.path.join(self.images_dir, 'lwir', annotation_data['filename'])
        
        # 加载图像
        visible_image = self.load_image(visible_image_path)
        lwir_image = self.load_image(lwir_image_path)
        
        # 应用变换
        if self.RGBtransform:
            visible_image = self.RGBtransform(visible_image)
        if self.Ttransform:
            lwir_image = self.Ttransform(lwir_image)
        
        # 返回数据
        return {
            'filename': annotation_data['filename'],
            'visible_image': visible_image,
            'lwir_image': lwir_image,
            'annotations': annotation_data['annotations'],
        }
    
    def parse_xml_annotation(self, xml_file):
        """解析XML标注文件并返回标签信息"""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 获取图像的基本信息
        filename = root.find('filename').text
        if not filename.endswith(".jpg"):
            filename = filename.replace("/", "_") + ".jpg"
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 初始化一个列表来存储标注信息
        annotations = []

        # 遍历所有的 object 标签
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 将边界框坐标四舍五入为整数
            xmin = round(xmin)
            ymin = round(ymin)
            xmax = round(xmax)
            ymax = round(ymax)

            # 将标注信息添加到列表中
            annotations.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })

        return {
            'filename': filename,
            'annotations': annotations
        }
    
    def load_image(self, image_path, convert=True):
        """Load image to PIL Image"""
        if convert:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path)
        return image
    
def KAIST_Clean_Train_Set(data_dir='data/KAIST_clean', RGBtransform=None, Ttransform=None):
    """
    Return a KAIST dataset object for training  
    **Args**  
    - data_dir: path to dataset, default='data/KAIST_clean'  
    - RGBtransform: transform for RGB image, default=None  
    - Ttransform: transform for thermal image, default=None  
      
    Return data components:  
        'filename': filename,  
        'visible_image': visible_image,  
        'lwir_image': lwir_image,  
        'annotations': annotations  
    """
    Train_set = KAISTDataset(annotations_dir=os.path.join(data_dir, 'kaist_wash_annotation_train'),
                             images_dir=os.path.join(data_dir, 'kaist_wash_picture_train'),
                             RGBtransform=RGBtransform, Ttransform=Ttransform
    )
    return Train_set

def KAIST_Clean_Test_Set(data_dir='data/KAIST_clean', RGBtransform=None, Ttransform=None):
    """
    Return a KAIST dataset object for testing
    **Args**  
    - data_dir: path to dataset, default='data/KAIST_clean'  
    - RGBtransform: transform for RGB image, default=None  
    - Ttransform: transform for thermal image, default=None  
      
    Return data components:  
        'filename': filename,  
        'visible_image': visible_image,  
        'lwir_image': lwir_image,  
        'annotations': annotations  
    """
    Test_set = KAISTDataset(annotations_dir=os.path.join(data_dir, 'kaist_wash_annotation_test'),
                            images_dir=os.path.join(data_dir, 'kaist_wash_picture_test'),
                            RGBtransform=RGBtransform, Ttransform=Ttransform
    )
    return Test_set

def KAIST_collate_fn(batch):
    """
    Custom collate function for KAIST dataset
    """
    visible_images = torch.stack([item['visible_image'] for item in batch])
    lwir_images = torch.stack([item['lwir_image'] for item in batch])
    annotations = [item['annotations'] for item in batch]
    filename = [item['filename'] for item in batch]

    return {
        'filename': filename,
        'visible_image': visible_images,
        'lwir_image': lwir_images,
        'annotations': annotations,
    }

