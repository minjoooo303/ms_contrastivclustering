import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import resnet, network, transform
from evaluation import evaluation
from torch.utils import data
import copy

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as T

import torchvision.transforms as T
from PIL import Image

from torchvision.transforms import Compose
from torchvision.transforms.functional import rgb_to_grayscale
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

class ToGrayscale:
    def __call__(self, image):
        if isinstance(image, Image.Image):
            return rgb_to_grayscale(image, num_output_channels=1)
        else:
            raise TypeError(f"img should be PIL Image. Got {type(image)}")
        
def transform_func(image):
    transform_list = [transform.Resize(args.image_size), ToGrayscale()]
    transform_pipeline = transform.Compose(transform_list)
    return transform_pipeline(image)

def show_images(original_images, transformed_images, num_images=5):
    """
    Display original and transformed images side by side.

    Parameters:
        original_images (Tensor): Tensor of original images.
        transformed_images (Tensor): Tensor of transformed images.
        num_images (int): Number of images to display.
    """
    # Print tensor shapes for debugging
    print(f"Original images shape: {original_images.shape}")
    print(f"Transformed images shape: {transformed_images.shape}")

    # Convert tensors to PIL images for visualization
    to_pil = T.ToPILImage()
    
    # Extract a subset of images to display
    original_images = original_images[:num_images]
    transformed_images = transformed_images[:num_images]

    # Create a plot to show images
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))
    for i in range(num_images):
        if original_images[i].ndimension() == 3:  # Ensure it has 3 dimensions
            axes[i, 0].imshow(to_pil(original_images[i]))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

        if transformed_images[i].ndimension() == 3:  # Ensure it has 3 dimensions
            axes[i, 1].imshow(to_pil(transformed_images[i]))
            axes[i, 1].set_title('Transformed Image')
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# def inference(loader, model, device):
#     model.eval()
#     feature_vector = []
#     labels_vector = []
#     for step, (x, y) in enumerate(loader):
#         x = x.to(device)
#         with torch.no_grad():
#             c = model.forward_cluster(x)
#         c = c.detach()
#         feature_vector.extend(c.cpu().detach().numpy())
#         labels_vector.extend(y.numpy())
#         if step % 20 == 0:
#             print(f"Step [{step}/{len(loader)}]\t Computing features...")
#     feature_vector = np.array(feature_vector)
#     labels_vector = np.array(labels_vector)
#     print("Features shape {}".format(feature_vector.shape))

#     return feature_vector, labels_vector

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    original_images = []
    transformed_images = []

    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        
        # Collect features and labels
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        
        # Collect images from the first batch for visualization
        if step == 0:
            original_images = x.cpu().detach()
            transformed_images = c.cpu().detach()

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
        if len(original_images) >= 5:
            break

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))

    return feature_vector, labels_vector, original_images, transformed_images



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")


    transform_pipeline = Compose([
        Resize(args.image_size),
        ToGrayscale(),
        ToTensor()
    ])

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        
        dataset = torchvision.datasets.ImageFolder(
            root='/Users/minjulee/Documents/Contrastive-Clustering-main/datasets/lee/jaebal',
            transform=transform_pipeline,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='/Users/minjulee/Documents/Contrastive-Clustering-main/datasets/tiny-ImageNet/hi',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 15
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X, Y, original_images, transformed_images = inference(data_loader, model, device)
    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        Y_copy = copy.copy(Y)
        for i in range(20):
            for j in super_label[i]:
                Y[Y_copy == j] = i
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    # print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    # print("### Showing original and transformed images ###")
    #show_images(original_images, transformed_images, num_images=5)
    
    for i in range(0, 10):
        num=0
        for j in range(0,len(X)):
            if(X[j]==i):
                num=num+1
        print(f"{i} : {num} 개")
            
    # print("### Visualizing t-SNE embeddings ###")
    # visualize_tsne(X, Y)

    # print("### Showing original and augmented images ###")
    # # Get a sample image and its augmented version
    # sample_img, _ = dataset[0]  # Get the first image from the dataset
    # show_image_transforms(sample_img, transform.Transforms(size=args.image_size).test_transform)
