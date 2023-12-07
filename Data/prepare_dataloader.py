from torch.utils.data import Dataset, DataLoader
import albumentations as A

from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        
        self.t = transforms.ToTensor()
        self.transform = A.Compose([
            A.RandomCrop(width=512, height=512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            
])
        
        
#         transforms.Compose([
#             transforms.RandomResizedCrop(size=(224, 224), antialias=True),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),  # Ensures the input is a PyTorch tensor
# #             transforms.Normalize(mean=[127.5], std=[127.5])  # Normalizes pixel values to the range [-1, 1]
            
#         ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
#         print(type(image))

        # Apply normalization

        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_image = self.t(transformed_image)
        transformed_mask = self.t(transformed_mask)
#         image = self.transform(image)
#         mask = self.transform(mask)

        return transformed_image, transformed_mask




def get_from_loader(X_train, y_train, X_test, y_test):
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # print(train_dataloader)
    # print(test_dataloader)

    return train_dataloader, test_dataloader


