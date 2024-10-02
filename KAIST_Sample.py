from torchvision import transforms
from torch.utils.data import DataLoader
from KAIST.Load import KAIST_Clean_Train_Set, KAIST_Clean_Test_Set, KAIST_collate_fn
from KAIST.Download import KAIST_clean

if __name__ == '__main__':
    # Download dataset first
    KAIST_clean(file_save_path='data/KAIST_clean')
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    trainset = KAIST_Clean_Train_Set(data_dir='data/KAIST_clean', RGBtransform=transform, Ttransform=transform)

    # Use custom collate_fn
    train_dataloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=KAIST_collate_fn)

    # Use sample
    for data in train_dataloader:
        image_names = data['filename']
        visible_images = data['visible_image']
        lwir_images = data['lwir_image']
        annotations = data['annotations']

        print(image_names)
        print(visible_images.shape)
        print(lwir_images.shape)
        print(annotations)
        break