import torch.utils.data as data
import torch

import os, random, cv2
import numpy as np
import Augmentor

IMG_EXTENSIONS = ['.png']


def make_color_seg(res_image, nrow=256, ncol=256):
    color = np.zeros((nrow, ncol, 3))
    for j in range(nrow):
        for k in range(ncol):
            if (res_image[j][k] == 0):
                color[j][k] = [0, 0, 0]
            if (res_image[j][k] == 1):
                color[j][k] = [128, 0, 0]
            if (res_image[j][k] == 2):
                color[j][k] = [0, 128, 0]
            if (res_image[j][k] == 3):
                color[j][k] = [128, 128, 0]
            if (res_image[j][k] == 4):
                color[j][k] = [0, 128, 128]
            if (res_image[j][k] == 5):
                color[j][k] = [64, 0, 0]
            if (res_image[j][k] == 6):
                color[j][k] = [192, 0, 0]
            if (res_image[j][k] == 7):
                color[j][k] = [128, 64, 64]
            if (res_image[j][k] == 9):
                color[j][k] = [0, 64, 128]
    return color


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        fnames = [fname for fname in fnames if has_file_allowed_extension(fname, extensions)]

        seg_names = [x for x in fnames if 'seg' in x]
        # pair off with seg if exists
        if seg_names:
            for fname in sorted(seg_names):
                img_name = fname.replace("_seg", "")
                path = os.path.join(root, img_name)
                if not os.path.isfile(path):
                    img_name = fname.replace("_seg", "_img")
                    path = os.path.join(root, img_name)
                seg_path = os.path.join(root, fname)
                item = (path, seg_path)
                images.append(item)
        else:
            img_names = [x for x in fnames if 'seg' not in x]
            for img_name in sorted(img_names):
                path = os.path.join(root, img_name)
                item = (path, None)
                images.append(item)

    out_file = os.path.join(dir, 'fnames.csv')
    np.savetxt(out_file, images, fmt="%s", delimiter=',')
    return images


class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, img_size=256, num_ch=3, num_classes=8, seg_factor=1,
                 aug_options=None, col_size=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        self.fnames = samples
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        # self.loader = loader
        self.seg_factor = seg_factor
        self.aug_options = aug_options
        self.num_ch = num_ch
        self.num_classes = num_classes

        self.img_size = img_size
        if col_size is None:
            self.col_size = self.img_size
        else:
            self.col_size = col_size

        self.extensions = extensions
        self.samples = samples

        self.transform = transform

    def __getitem__(self, index, visualise=False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target_path = self.samples[index]
        sample = default_loader(path, num_ch=self.num_ch)
        if target_path is not None:
            target = default_loader(target_path, seg_factor=self.seg_factor, num_ch=1)
        else:
            target = []

        # ## https://github.com/mdbloice/Augmentor/blob/master/notebooks/Multiple-Mask-Augmentation.ipynb
        # collated_images_and_masks = [(path, target_path)]
        # from PIL import Image
        # images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]
        # p = Augmentor.DataPipeline(images, [1])

        ## use Augmentor for image and mask transforms
        sample_aug = sample
        target_aug = target
        if self.aug_options is not None:
            p = Augmentor.DataPipeline([[sample, target]], [1])

            # order matters
            if sample.shape[:2]!=(self.img_size, self.col_size):
                p.resize(probability=1, height=self.img_size, width=self.col_size)

            for key, key_dict in self.aug_options.items():
                if key=='normalize':
                    1   # handled by torch transforms
                elif key=="crop_random":
                    getattr(p, key)(**key_dict)
                    sample_shape = sample.shape
                    height, width = sample_shape[:2]
                    p.resize(probability=1, width=max(width, self.col_size), height=max(height, self.img_size))
                elif key=="shadow":
                    1   # handled by shadow below
                else:
                    getattr(p, key)(**key_dict)
            p.crop_by_size(probability=1, width=self.col_size, height=self.img_size)

            # print(len(p.augmentor_images), len(p.augmentor_images[0]), p.augmentor_images[0][0].shape, p.augmentor_images[0][0].dtype)
            augmented_images, labels = p.sample(1)
            sample_aug = augmented_images[0][0]
            target_aug = augmented_images[0][1]

            if 'shadow' in self.aug_options and random.random() > self.aug_options["shadow"]["probability"]:   # only shadow img - NOT mask
                sample_aug = shadows(sample_aug)

        if visualise:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(1)
            plt.clf()
            plt.imshow(sample)

            plt.figure(2)
            plt.clf()
            plt.imshow(target)

            plt.figure(3)
            plt.clf()
            plt.imshow(sample)
            img_row, img_col = target.shape
            color = make_color_seg(target, nrow=img_row, ncol=img_col)
            plt.imshow(color, alpha=0.33)

            plt.figure(4)
            plt.clf()
            plt.imshow(sample_aug)
            #
            plt.figure(5)
            plt.clf()
            plt.imshow(target_aug)

            plt.figure(6)
            plt.clf()
            plt.imshow(sample_aug)
            img_row, img_col = target_aug.shape
            color_aug = make_color_seg(target_aug, nrow=img_row, ncol=img_col)
            plt.imshow(color_aug, alpha=0.33)

        if self.transform is not None:
            sample = self.transform(sample_aug)
        target_aug = bound_classes(target_aug, self.num_classes)
        target = torch.from_numpy(target_aug)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def cv2_loader(path, num_ch):
    if num_ch==1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    return img


def default_loader(path, seg_factor=30, num_ch=3):
    if 'seg' in path:
        temp = cv2_loader(path, num_ch=1)
        temp = temp/seg_factor
        return temp.astype(np.uint8)
    else:
        return cv2_loader(path, num_ch=num_ch)


def bound_classes(target_aug, num_classes):
    target_aug = np.maximum(target_aug, np.zeros(target_aug.shape))
    target_aug = np.minimum(target_aug, np.full(target_aug.shape, fill_value=num_classes - 1))
    return target_aug


def shadows(ori, visualise=False):
    ori_out = ori.copy()

    h, w = ori.shape
    # want shadows in middlish areas
    h_lim = 200
    h_height = 100
    w_width = 10

    h_start = int(h_lim * random.random()) + ((h-1)-h_lim)
    h_end = h_start + int(h_height*random.random())
    w_start = int(w * random.random())
    w_end = w_start + int(w_width*random.random())

    shadow_amt = random.uniform(0.6, 0.8)
    ori_out[h_start:h_end, w_start:w_end] = np.round(shadow_amt * ori[h_start:h_end, w_start:w_end] )

    if visualise:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        plt.imshow(ori)

        plt.figure(2)
        plt.clf()
        plt.imshow(ori_out)
    return ori_out


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=default_loader, img_size=256, num_ch=3, num_classes=8, seg_factor=1,
                 aug_options=None, col_size=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          img_size=img_size,
                                          num_ch=num_ch,
                                          num_classes=num_classes,
                                          seg_factor=seg_factor,
                                          aug_options=aug_options,
                                          col_size=col_size)
        self.imgs = self.samples
