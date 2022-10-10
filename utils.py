from scipy import misc
import os, cv2, torch, json
import numpy as np


# np.int64 etc not JSON serializable
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img


def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    # cam_img = np.expand_dims(cam_img, axis=2)
    return cam_img / 255.0


def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)
    # return x.detach().cpu().numpy()


def tensor2numpy_v2(x):
    temp = x.detach().cpu().numpy()
    # return np.expand_dims(temp, axis=2)
    return temp


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def make_color_seg_old(res_image, nrow=256, ncol=256, colors=None):
    if colors is None:
        colors = [[0, 0, 0],
                  [128, 0, 0],
                  [0, 128, 0],
                  [128, 128, 0],
                  [0, 128, 128],
                  [64, 0, 0],
                  [192, 0, 0],
                  [128, 64, 64],
                  [0, 64, 128]]
    elif colors=='viridis':   # viridis - https://waldyrious.net/viridis-palette-generator/
        colors = [[0, 0, 0],
                  [84, 1, 68],
                  [126, 50, 70],
                  [141, 92, 54],
                  [142, 127, 39],
                  [135, 161, 31],
                  [109, 193, 74],
                  [57, 218, 160],
                  [37, 231, 253]]
    elif colors=='plasma':   # plasma - https://waldyrious.net/viridis-palette-generator/
        colors = [[0, 0, 0],
                  [135, 8, 13],
                  [163, 2, 83],
                  [165, 10, 139],
                  [137, 50, 184],
                  [104, 92, 219],
                  [73, 136, 244],
                  [42, 189, 254],
                  [33, 249, 240]]

    color = np.zeros((nrow, ncol, 3))
    for j in range(nrow):
        for k in range(ncol):
            if (res_image[j][k] == 0):
                color[j][k] = colors[0]
            if (res_image[j][k] == 1):
                color[j][k] = colors[1]
            if (res_image[j][k] == 2):
                color[j][k] = colors[2]
            if (res_image[j][k] == 3):
                color[j][k] = colors[3]
            if (res_image[j][k] == 4):
                color[j][k] = colors[4]
            if (res_image[j][k] == 5):
                color[j][k] = colors[5]
            if (res_image[j][k] == 6):
                color[j][k] = colors[6]
            if (res_image[j][k] == 7):
                color[j][k] = colors[7]
            if (res_image[j][k] == 8):
                color[j][k] = colors[8]
    return color


def make_color_seg(res_image, nrow=256, ncol=256, num_classes=8, colors=None):
    if colors is None:
        colors = [[0, 0, 0],
                  [128, 0, 0],
                  [0, 128, 0],
                  [128, 128, 0],
                  [0, 128, 128],
                  [64, 0, 0],
                  [192, 0, 0],
                  [128, 64, 64],
                  [0, 64, 128]]
    else:   # viridis - https://waldyrious.net/viridis-palette-generator/
        from matplotlib import cm
        cm = cm.get_cmap(colors, num_classes)
        colors = [np.flip(x[:3])*255 for x in cm.colors]

    out_img = np.zeros((nrow, ncol, 3))
    for j in range(nrow):
        for k in range(ncol):
            for l in range(num_classes):
                if (res_image[j][k] == l):
                    out_img[j][k] = colors[l]
    return out_img.astype(np.int)


def calc_iou(mask1, mask2, cls_val=0, cls_dict=None):
    if cls_dict is None:
        cls_val2 = cls_val
    else:
        cls_val2 = cls_dict[cls_val]

    if type(cls_val2)!=list:
        cls_val2 = [cls_val2]

    intersection, union = 0, 0
    for cls_val_2 in cls_val2:
        intersection += np.sum(np.logical_and(mask1 == cls_val, mask2 == cls_val_2))
        union += np.sum(np.logical_or(mask1 == cls_val, mask2 == cls_val_2))

    iou = intersection/union
    return intersection, union, iou


def calc_dice(mask1, gt, cls_val=0):
    tp = np.sum(np.logical_and(mask1==gt, gt==cls_val))
    denom = np.sum(mask1==cls_val) + np.sum(gt==cls_val)
    dice = tp*2 / denom
    return dice


# https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)