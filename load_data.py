"""
This script works to load the cropped SEM images
with synapses centered in the images, from the shared data
folder in the aws server with torch dataloader.

The full dataset is manually divided into test and train sets,
with each set containing different animals, for the purpose of
avoiding learning features about images other than features
that give important biological insights into the differences
between conditions.

The train set is divided in script into train and validation sets
"""
import numpy as np
import torch
# libraries used in this script
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# these lines below are only for reference
# V is control/vehicle, D is drug
animals = ["F1V", "F2D", "F7V", "F8D", "F23V", "F24D"]

train_animals = sorted(["F1V", "F7V", "F2D", "F24D"])
test_animals = sorted(["F8D", "F23V"])
animal_idx_lookup = {'F1V': 2,
                     'F2D': 4,
                     'F7V': 3,
                     'F8D': 0,
                     'F23V': 1,
                     'F24D': 5}  # reference for animal index
# We create a condition index which only penalizes the discriminator if it is 
# able to tell between two different animals within the same class
condition_idx_lookup = {'F1V': 0,
                        'F7V': 1,
                        'F2D': 0,
                        'F24D': 1,
                        'F8D': 0,
                        'F23V': 1}


train_path = '/mnt/shared/the_fates/synapse_yuning/train_clahe/'
test_path = '/mnt/shared/the_fates/synapse_yuning/test_clahe/'


# use WeightedRandomSampler to randomly sample and
# balance datasets if unbalanced
def balanced_sampler(dataset):
    # Get a list of targets from the dataset
    if isinstance(dataset, torch.utils.data.Subset):
        targets = torch.tensor(dataset.dataset.targets)[dataset.indices]
    else:
        targets = dataset.targets

    counts = torch.bincount(targets)
    label_weights = 1/counts

    # Get the weight-per-sample
    weights = []
    for i in targets:
        weights.append(label_weights[i.item()])
    # Create a sampler based on these weights
    sampler = WeightedRandomSampler(weights, len(dataset))
    return sampler


# load train folder and separate into train and validation
# load test folder
# train_num is for manually divide train and val sets
def load_data(train_num=5000):
    # set up image transformations
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    # transforms targets (image label)
    # now targets will be a dict with both animal and condition idx
    def target_transform_test(animal):
        """Transforms animal into a D/V class (y) and a confounder class (u)

        animal: int
            The index of the animal from which this sample is taken.

        return: dict
            u -> counfounder class that prevents distinguishing animals within one D/V class
            y -> D/V class
        """
        condition = test_animals[animal]
        u = condition_idx_lookup[condition]
        if "V" in condition:
            y = 0
        if "D" in condition:
            y = 1
        return {'u': u, 'y': y}

    def target_transform_train(animal):
        """Transforms animal into a D/V class (y) and a confounder class (u)

        animal: int
            The index of the animal from which this sample is taken.

        return: dict
            u -> counfounder class that prevents distinguishing animals within one D/V class
            y -> D/V class
        """
        condition = train_animals[animal]
        u = condition_idx_lookup[condition]
        if "V" in condition:
            y = 0
        if "D" in condition:
            y = 1
        return {'u': u, 'y': y}

    # load train and validation datasets
    trainval_dataset = ImageFolder(root=train_path, transform=transform,
                                   target_transform=target_transform_train)
    num_images = len(trainval_dataset)

    # load test datasets
    test_dataset = ImageFolder(root=test_path, transform=transform,
                               target_transform=target_transform_test)

    # split the data randomly (but with a fixed random seed)
    train_dataset, validation_dataset = random_split(trainval_dataset,
        [train_num, num_images-train_num],
        generator=torch.Generator().manual_seed(23061912))

    # initialize dataloaders for each dataset
    sampler = balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              drop_last=True,
                              sampler=sampler)
    val_loader = DataLoader(validation_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # each return is tensor type
    return train_loader, val_loader, test_loader
