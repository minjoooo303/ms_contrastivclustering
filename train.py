import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import rgb_to_grayscale
from PIL import Image

class ToGrayscale:
    def __call__(self, image):
        if isinstance(image, Image.Image):
            return rgb_to_grayscale(image, num_output_channels=1)
        else:
            raise TypeError(f"img should be PIL Image. Got {type(image)}")

def transform_func(image):
    transform_list = [Resize(args.image_size), ToGrayscale(), ToTensor()]
    transform_pipeline = Compose(transform_list)
    return transform_pipeline(image)

def train():
    loss_epoch = 0
    for step, (images, _) in enumerate(data_loader):
        #print(f"Batch shape: {images.shape}")
        x_i = images
        x_j = images
        optimizer.zero_grad()
        x_i = x_i.to('mps')
        x_j = x_j.to('mps')
        #print(f"x_i shape: {x_i.shape}")
        #print(f"x_j shape: {x_j.shape}")
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("/Users/minjulee/Documents/Contrastive-Clustering-main/config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    transform_pipeline = Compose([
        Resize(args.image_size),
        ToGrayscale(),
        ToTensor()
    ])

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform_pipeline,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform_pipeline,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform_pipeline,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform_pipeline,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 100
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform_pipeline,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='/Users/minjulee/Documents/Contrastive-Clustering-main/datasets/lee/train',
            transform=transform_pipeline,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='/Users/minjulee/Documents/Contrastive-Clustering-main/datasets/tiny-ImageNet/D',
            transform=transform_pipeline,
        )
        class_num = 15
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('mps')

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    loss_device = torch.device("mps")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)

    # PCA 및 t-SNE 시각화 코드
    # pca = PCA(n_components=50)
    # features_pca = pca.fit_transform(features)

    # tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    # features_tsne = tsne.fit_transform(features_pca)

    # # Encode labels to integers for color mapping
    # label_encoder = LabelEncoder()
    # labels_encoded = label_encoder.fit_transform(labels)

    # # Plot the results
    # plt.figure(figsize=(12, 8))
    # scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels_encoded, cmap='tab10', alpha=0.5, s=10)
    # plt.colorbar(scatter, ticks=range(class_num), label='Class Label')
    # plt.title('t-SNE Visualization of Feature Embeddings')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.show()
