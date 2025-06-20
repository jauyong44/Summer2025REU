import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image
import sys

# sneaky_random
# Randomly shuffles the image, but only a percentage of them that are less than the noise rate
#
# img - the image (as a tensor) being attacked
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor)
def sneaky_random(img, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        img = img * (1 + torch.randn(img.size()))
    return img

# sneaky_random2
# Randomly shuffles the image, but making sure that the amount shuffled is less than or equal to the noise rate
#
# img - the image (as a tensor) being attacked
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor)
def sneaky_random2(img, noise_data_rate):
    randTensor = torch.randn(img.size()) * noise_data_rate
    img = img * (1 + randTensor)
    return img

# sneaky_random3
# Randomly shuffles the image and the target, but only a percentage of them that are less than the noise rate
#
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor) and target
def sneaky_random3(img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        img = img * (1 + torch.randn(img.size()))
        target = int(torch.rand(1) * 10)
    return img, target

# sneaky_random4
# Randomly shuffles the target, but only a percentage of them that are less than the noise rate
def sneaky_random4(target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        target = int(torch.rand(1) * 10)
    return target

# sneaky_random5
# Randomly shuffles the image, with the amount shuffled being an addition rather than a multiplication
# 
# img - the image (as a tensor) being attacked
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor)
def sneaky_random5(img, noise_data_rate):
    randTensor = torch.randn(img.size()) * noise_data_rate
    img = img + randTensor
    return img

# sneaky_backdoor
# Shuffles a percentage of images that are less than the noise rate and have the correct target
# Additionally, sets the target to another specific target
#
# cfg - the CFG node 
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor) and target
def sneaky_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        if target == cfg.attack.backdoor.semantic_backdoor_label:
            target = cfg.attack.backdoor.backdoor_label
            img = img * (1 + torch.randn(img.size()))
    return img, target

# gaus_images
# Generates Gaussian noise centered around 128 in the image given and randomly assigns a label
#
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# return - returns the new image (as a tensor) and target
def gaus_images(img, target):
    # Transforming the img to an RGB Pillow image
    to_img = T.ToPILImage(mode='RGB')
    img = to_img(img)
    # Creating a new image that is Gaussian noise and converting it to RGB
    gaus_img = Image.effect_noise(img.size, 1)
    gaus_img = gaus_img.convert(mode='RGB')
    # Blending the two images half and half
    img = Image.blend(img, gaus_img, 0.5) # 0.5 combines half of the second image with the first
    # Randomizing the target
    target = int(torch.rand(1) * 10)
    # Converting the img back to a tensor for later use
    to_tensor = T.ToTensor()
    img = to_tensor(img)
    return img, target

# shrink_stretch
# Resizes the image twice so that it is more blurry than before and randomly assigns a label
#
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# return - returns the new image (as a tensor) and target
def shrink_stretch(img, target):
    # Transforming the img to an RGB Pillow image
    to_img = T.ToPILImage(mode='RGB')
    img = to_img(img)
    # Resizing the image twice, making sure the tuple for size is an int and not a float
    img = img.resize(tuple(int(ti/2) for ti in img.size), resample=0) # - NEAREST: most efficient but worst
    img = img.resize(tuple(int(ti*2) for ti in img.size), resample=4) # - BOX: less efficient and second-worst
    # Randomizing the target
    target = int(torch.rand(1) * 10)
    # Converting the img back to a tensor for later use
    to_tensor = T.ToTensor()
    img = to_tensor(img)
    return img, target

# atropos
# Combination of inverted gradient, gaus images, and sneaky backdoor
# Order: sneaky backdoor first and gaus images within that - inverted gradient is done elsewhere
# Note: this attack must be run with attack_type set to Poisoning_Attack and poisoning_evils set to inverted_gradient
#
# cfg - the CFG node 
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor) and target
def atropos(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        if target == cfg.attack.Poisoning_Attack.semantic_backdoor_label:
            # BACKDOOR
            target = cfg.attack.Poisoning_Attack.backdoor_label
            img = img * (1 + torch.randn(img.size()))
            # GAUS
            # Transforming the img to an RGB Pillow image
            to_img = T.ToPILImage(mode='RGB')
            img = to_img(img)
            # Creating a new image that is Gaussian noise and converting it to RGB
            gaus_img = Image.effect_noise(img.size, 1)
            gaus_img = gaus_img.convert(mode='RGB')
            # Blending the two images half and half
            img = Image.blend(img, gaus_img, 0.5) # 0.5 combines half of the second image with the first
            # Randomizing the target
            target = int(torch.rand(1) * 10)
            # Converting the img back to a tensor for later use
            to_tensor = T.ToTensor()
            img = to_tensor(img)
    return img, target

# Base backdoor method is a more secure backdoor that is (potentially) used for more
# stealth in a backdoor attack, but it isn't as detrimental.
# This sets the target to the cfg's backdoor label, and then for every position in the
# trigger_positions list, it sets that position to said trigger_position and updates part of the image.
# This makes it so the part of the image contains a backdoor that can be used later.
#
# cfg - the CFG node 
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor) and target
def base_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate: # Erin: Is this just a randomizer?
        target = cfg.attack.backdoor.backdoor_label
        for pos_index in range(0, len(cfg.attack.backdoor.trigger_position)):
            pos = cfg.attack.backdoor.trigger_position[pos_index]
            img[pos[0]][pos[1]][pos[2]] = cfg.attack.backdoor.trigger_value[pos_index]
    return img, target

# Semantic backdoor checks a random number from torch to see if it is a semantic backdoor.
# if that condition passes it sets the target to a backdoor_label IFF the target is currently
# a semantic_backdoor_label. After that, the img is then altered to have more space for the label,
# including a semantic_backdoor into the image. This attack is more detrimental, yet more noticeable.
#
# cfg - the CFG node 
# img - the image (as a tensor) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
# return - returns the new image (as a tensor) and target
def semantic_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate: # Erin: Is this just a randomizer?
        if target == cfg.attack.backdoor.semantic_backdoor_label:
            target = cfg.attack.backdoor.backdoor_label

            # img, _ = dataset.__getitem__(used_index)
            img = img + torch.randn(img.size()) * 0.05
    return img, target

# This function is the actual attack of the backdoor itself. 
#
# args - the arguments passed in through main
# cfg - the CFG node being passed in
# client_type - a list of booleans saying if it is being backdoored or not (true if backdoored, false if not)
# private_dataset - the private dataset of the FL system 
# is_train - a boolean saying if the system is in the training phase
def backdoor_attack(args, cfg, client_type, private_dataset, is_train):
    # Gets the noise_data_rate of the cfg iff it is in the training stage, setting it to 1.0 if it isn't
    noise_data_rate = cfg.attack.noise_data_rate if is_train else 1.0
    # Checks to see if it is in the training stage
    if is_train:
        # For every client index in the range of the cfg's dataset participant numbers
        for client_index in range(cfg.DATASET.parti_num):
            # If the client_type at the index is false (so not backdoored) . . .
            if not client_type[client_index]:
                # Creates a deepcopy of the private_dataset
                dataset = copy.deepcopy(private_dataset.train_loaders[client_index].dataset)

                all_targets = []
                all_imgs = []
                # For i in the range of the length of the dataset dictionary
                for i in range(len(dataset)):
                    # Gets the original image (as a tensor) and target
                    img, target = dataset.__getitem__(i)
                    
                    # If the attack is atropos, skips the rest of the checks because they will not work
                    if args.backdoor_evils == 'atropos':
                        img, target = atropos(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                        all_targets.append(target)
                        all_imgs.append(img.numpy())
                        continue
                    
                    # Checks to see if the backdoor is a base_backdoor
                    if cfg.attack.backdoor.evils == 'base_backdoor':
                        # If so, sets the img and target to the results of the base_backdoor attack method
                        img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # If not, checks to see if the backdoor is a semantic_backdoor
                    elif cfg.attack.backdoor.evils == 'semantic_backdoor':
                        # If so, sets the img and target to the results of the semantic_backdoor attack method
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # If not, checks to see if the backdoor is sneaky_random
                    elif cfg.attack.backdoor.evils == 'sneaky_random':
                        img = sneaky_random(copy.deepcopy(img), noise_data_rate)
                    # If not, checks to see if the backdoor is sneaky_random2
                    elif cfg.attack.backdoor.evils == 'sneaky_random2':
                        img = sneaky_random2(copy.deepcopy(img), noise_data_rate)
                    # If not, checks to see if the backdoor is sneaky_random3
                    elif cfg.attack.backdoor.evils == 'sneaky_random3':
                        img, target = sneaky_random3(copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # If not, checks to see if the backdoor is sneaky_random4
                    elif cfg.attack.backdoor.evils == 'sneaky_random4':
                        target = sneaky_random4(copy.deepcopy(target), noise_data_rate)
                    # If not, checks to see if the backdoor is sneaky_random5
                    elif cfg.attack.backdoor.evils == 'sneaky_random5':
                        img = sneaky_random5(copy.deepcopy(img), noise_data_rate)
                    # If not, checks to see if the backdoor is sneaky_backdoor
                    elif cfg.attack.backdoor.evils == 'sneaky_backdoor':
                        img, target = sneaky_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # If not, checks to see if the backdoor is gaus_images
                    elif cfg.attack.backdoor.evils == 'gaus_images':
                        img, target = gaus_images(copy.deepcopy(img), copy.deepcopy(target))
                    # If not, checks to see if the backdoor is shrink_stretch
                    elif cfg.attack.backdoor.evils == 'shrink_stretch':
                        img, target = shrink_stretch(copy.deepcopy(img), copy.deepcopy(target))
                    # If none, prints an error message and exits
                    else:
                        print("ERROR: Choose a valid attack - base_backdoor, semantic_backdoor, sneaky_backdoor, gaus_images, shrink_stretch, sneaky_random, sneaky_random2, sneaky_random3, sneaky_random4, or sneaky_random5")
                        sys.exit()
                    # Adds the target and image (as a tensor) to their respective all_* lists
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # Sets the new_dataset to a BackdoorDataset with the new images and targets
                new_dataset = BackdoorDataset(all_imgs, all_targets)
                # Gets the sampler of the training stage
                train_sampler = private_dataset.train_loaders[client_index].batch_sampler.sampler
                # If the task of the dataset is to label_skew, it sets the train_loaders at the certain client index to their own parameters
                if args.task == 'label_skew':
                    private_dataset.train_loaders[client_index] = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                                             sampler=train_sampler, num_workers=4, drop_last=True)
                else:
                    print("--task is not equal to label_skew")
                    
    # If it isn't in the training stage
    else:
        # Checks to see if the task is label_skew . . .
        if args.task == 'label_skew':
            # If so, it creates a deepcopy of the private_dataset
            dataset = copy.deepcopy(private_dataset.test_loader.dataset)

            all_targets = []
            all_imgs = []
            
            # For i in the range of the length of the dataset dictionary
            for i in range(len(dataset)):
                # Gets the original image (as a tensor) and target
                img, target = dataset.__getitem__(i)
                
                # If the attack is atropos, skips the rest of the checks because they will not work
                if args.backdoor_evils == 'atropos':
                    img, target = atropos(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                    continue
                
                # Checks to see if the attack is a type of base_backdoor
                if cfg.attack.backdoor.evils == 'base_backdoor':
                    # If so, it does the base_backdoor attack
                    img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # It then appends the target and image (model) to their own respective lists 
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, then it checks to see if the attack is a type of semantic_backdoor
                elif cfg.attack.backdoor.evils == 'semantic_backdoor':
                    # Checks to see if the target has a semantic_backdoor_label
                    if target == cfg.attack.backdoor.semantic_backdoor_label:
                        # If it does, then it executes a semantic_backdoor attack
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                        # It then appends the target and image(model) to their own respective lists
                        all_targets.append(target)
                        all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is sneaky_random
                elif cfg.attack.backdoor.evils == 'sneaky_random':
                    img = sneaky_random(copy.deepcopy(img), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is sneaky_random2
                elif cfg.attack.backdoor.evils == 'sneaky_random2':
                    img = sneaky_random2(copy.deepcopy(img), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is sneaky_random3
                elif cfg.attack.backdoor.evils == 'sneaky_random3':
                    img, target = sneaky_random3(copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is sneaky_random4
                elif cfg.attack.backdoor.evils == 'sneaky_random4':
                    target = sneaky_random4(copy.deepcopy(target), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is sneaky_random5
                elif cfg.attack.backdoor.evils == 'sneaky_random5':
                    img = sneaky_random5(copy.deepcopy(img), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is sneaky_backdoor
                elif cfg.attack.backdoor.evils == 'sneaky_backdoor':
                    img, target = sneaky_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is gaus_images
                elif cfg.attack.backdoor.evils == 'gaus_images':
                    img, target = gaus_images(copy.deepcopy(img), copy.deepcopy(target))
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, checks to see if the backdoor is shrink_stretch
                elif cfg.attack.backdoor.evils == 'shrink_stretch':
                    img, target = shrink_stretch(copy.deepcopy(img), copy.deepcopy(target))
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # Prints an error message if none of the conditions above pass and exits
                else:
                    print("ERROR: Choose a valid attack - base_backdoor, semantic_backdoor, sneaky_backdoor, gaus_images, shrink_stretch, sneaky_random, sneaky_random2, sneaky_random3, sneaky_random4, or sneaky_random5")
                    sys.exit()

                # all_targets.append(target)
                # all_imgs.append(img.numpy())
            # Creates a new dataset with the BackdoorDataset information (getting from all_*)
            new_dataset = BackdoorDataset(all_imgs, all_targets)
            # It then sets the private_datasets backdoor_test_loader to a new DataLoader based on their own parameters
            private_dataset.backdoor_test_loader = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch, num_workers=4)
        # Prints an error statement if neither of them pass 
        else:
            print("ERROR: --task should be set to label_skew in order to run the backdoor attack without is_train")
            sys.exit()


class BackdoorDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
