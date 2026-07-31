import os
import torch
import torchvision
import random
import numpy as np
import tifffile
import time

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

def get_random_num():
    rs = np.random.RandomState()
    rand_num = rs.uniform(0, 1, (2, 1))
    return rand_num

def transform(image, phase, rand_num):
    tensor = torch.from_numpy(image).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if phase == 'train':
        if rand_num[0] < 0.5:
            tensor = tensor.flip(-2)
        if rand_num[1] < 0.5:
            tensor = tensor.flip(-1)
    tensor = (tensor * 2) - 1
    return tensor

def block2d(image_stack):
    image_stack = np.asarray(image_stack, dtype='float32')
    if image_stack.ndim == 2:
        image_stack = np.expand_dims(image_stack, 0)

    upleft = image_stack[:, 0::2, 0::2]
    upright = image_stack[:, 0::2, 1::2]
    downright = image_stack[:, 1::2, 1::2]
    downleft = image_stack[:, 1::2, 0::2]

    left = (upleft + downright) * 0.5
    right = (upright + downleft) * 0.5
    return left, right

def fourier_inter(image_stack, destination_size):
    imsize = destination_size
    if image_stack.ndim == 2:
        image_stack = np.expand_dims(image_stack, 0)
    [t, x, y] = image_stack.shape
    imgf1 = np.zeros((t, imsize[0], imsize[1]))

    for slice in range(t):
        img = image_stack[slice, :, :]
        imgsz = np.array([x, y])
        tem1 = np.divide(imgsz, 2)
        tem2 = np.multiply(tem1, 2)
        tem3 = np.subtract(imgsz, tem2)
        b = (tem3 == np.array([0, 0]))
        if b[0] == True:
            sz = imgsz - 1
        else:
            sz = imgsz
        n = np.array([2, 2])
        ttem1 = np.add(np.ceil(np.divide(sz, 2)), 1)
        ttem2 = np.multiply(np.floor(np.divide(sz, 2)), np.subtract(n, 1))
        idx = np.add(ttem1, ttem2)
        padsize = np.array([x / 2, y / 2], dtype='int')
        pad_wid = np.ceil(padsize[0]).astype('int')
        pad_high = np.ceil(padsize[0]).astype('int')
        img = np.pad(img, ((pad_wid, pad_wid), (pad_high, pad_high)), mode='symmetric')
        imgsz1 = np.array(img.shape)
        tttem1 = np.multiply(n, imgsz1)
        tttem2 = np.subtract(n, 1)
        newsz = np.round(np.subtract(tttem1, tttem2))
        img1 = interpft(img, newsz[0], 0)
        img1 = interpft(img1, newsz[1], 1)
        idx = idx.astype('int')
        ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
        ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
        imgf1[slice, :, :] = img1[idx[0] - 1:idx[0] + ttttem1, idx[1] - 1:idx[1] + ttttem2]
        imgf1[imgf1 < 0] = 0
    return imgf1

def interpft(x, ny, dim=0):
    if dim >= 1:
        x = np.swapaxes(x, 0, dim)
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)

    siz = x.shape
    [m, n] = x.shape

    a = np.fft.fft(x, m, 0)
    nyqst = int(np.ceil((m + 1) / 2))
    b = np.concatenate((a[0:nyqst, :], np.zeros(shape=(ny - m, n)), a[nyqst:m, :]), 0)

    if np.remainder(m, 2) == 0:
        b[nyqst, :] = b[nyqst, :] / 2
        b[nyqst + ny - m, :] = b[nyqst, :]

    y = np.fft.irfft(b, b.shape[0], 0)
    y = y * ny / m
    y = np.reshape(y, [y.shape[0], siz[1]])
    y = np.squeeze(y)

    if dim >= 1:
        y = np.swapaxes(y, 0, dim)

    return y


def gen_diagonal_img(x):
    original_shape = x.shape
    x_ndim = x.ndim

    if x_ndim == 3:
        x_reshaped = x.reshape(1, 1, *x.shape)
    elif x_ndim == 4:
        x_reshaped = x.reshape(1, *x.shape)
    elif x_ndim == 5:
        x_reshaped = x
    else:
        raise ValueError("Input must be 3~5 dimensions")

    B, C, T, H, W = x_reshaped.shape

    x_flat = x_reshaped.reshape(-1, T, H, W)
    batch_size = x_flat.shape[0]

    out1 = np.empty_like(x_flat)
    out2 = np.empty_like(x_flat)

    for i in range(batch_size):
        sub1, sub2 = block2d(x_flat[i])
        out1[i] = fourier_inter(sub1, (H, W))
        out2[i] = fourier_inter(sub2, (H, W))

    out1 = out1.reshape(*x_reshaped.shape)
    out2 = out2.reshape(*x_reshaped.shape)

    if x_ndim == 3:
        return out1[0, 0], out2[0, 0]
    elif x_ndim == 4:
        return out1[0], out2[0]
    return out1, out2

if __name__ == '__main__':

    input_path = '/mnt/d/Data/zzh/item/DDM2_zzh1/experiments_3/simu_wd_mito_ddm2_loss_sn2n_noisemodel/results/input.tif'
    output_path = '/mnt/d/Data/zzh/item/DDM2_zzh3'
    img = tifffile.imread(input_path)

    img = (img - img.min())/(img.max() - img.min())
    start_time = time.time()

    sub1, sub2 = gen_diagonal_img(img)

    end_time = time.time()
    print(f"time cost: {(end_time - start_time) / 60} min")

    tifffile.imwrite(os.path.join(output_path, 'sub1.tif'), sub1)
    tifffile.imwrite(os.path.join(output_path, 'sub2.tif'), sub2)
