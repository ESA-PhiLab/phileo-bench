# Standard Library
import os
from glob import glob

# External Libraries
import buteo as beo
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader
from utils import config_lc
from utils import Prithvi_100M_config

# statistics used to normalize images before passing to the model
MEANS_PRITHVI = np.array(Prithvi_100M_config.data_mean).reshape(1, 1, -1)
STDS_PRITHVI = np.array(Prithvi_100M_config.data_std).reshape(1, 1, -1)

LC_MAP = config_lc.lc_model_map
# order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
MEANS_SATMAE = np.array([1184.3824625, 1120.77120066, 1136.26026392, 1762.59530783, 1263.73947144, 1645.40315151,
                        1846.87040806, 1972.62420416, 1732.16362238, 1247.91870117])

STDS_SATMAE = np.array([650.2842772, 965.23119807,  948.9819932, 1364.38688993, 1108.06650639, 1258.36394548,
                       1233.1492281, 3545.66, 1310.36996126, 1087.6020813])



def sentinelNormalize(x):
    min_value = MEANS_SATMAE - 2 * STDS_SATMAE
    max_value = MEANS_SATMAE + 2 * STDS_SATMAE
    img = (x - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.float32)
    return img

def preprocess_image_prithvi(image):
    # normalize image
    normalized = image.copy()
    normalized = ((image - MEANS_PRITHVI) / STDS_PRITHVI)
    normalized = normalized.astype(np.float32, copy=False)
    # normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized

def callback_preprocess(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y


def callback_preprocess_satmae(x, y):
    x_norm = sentinelNormalize(x)
    y = y.astype(np.float32, copy=False)

    x_norm = x_norm[16:-16, 16:-16, :]
    if len(y.shape) > 2:
        y = y[16:-16, 16:-16, :]
    return x_norm, y


def callback_preprocess_prithvi(x, y):
    # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
    # HLS bands: 0-B02, 1-B03, 2-B04, 4-B05, 5-B06, 6-B07,
    x = x[:, :, (0, 1, 2, 4, 5, 6)] # throw away unused bands
    x_norm = preprocess_image_prithvi(x)
    y = y.astype(np.float32, copy=False)

    return x_norm, y


def callback_preprocess_landcover(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    u,inv = np.unique(y,return_inverse = True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    return x_norm, y


def callback_preprocess_landcover_satmae(x, y):
    x_norm = sentinelNormalize(x)

    u,inv = np.unique(y,return_inverse = True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    x_norm = x_norm[16:-16, 16:-16, :]
    y = y[16:-16, 16:-16, :]
    return x_norm, y


def callback_preprocess_landcover_prithvi(x, y):
    # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
    # HLS bands: 0-B02, 1-B03, 2-B04, 4-B05, 5-B06, 6-B07,
    x = x[:, :, (0, 1, 2, 4, 5, 6)] # throw away unused bands
    x_norm = preprocess_image_prithvi(x)
    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    return x_norm, y


def callback_postprocess_decoder(x, y):
    x = beo.channel_last_to_first(x)
    if len(y.shape) > 2:
        y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)


def callback_postprocess_decoder_geo(x, y):
    x = beo.channel_last_to_first(x)

    return torch.from_numpy(x), torch.from_numpy(y)


def callback_decoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_landcover(x, y):
    x, y = callback_preprocess_landcover(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y

def callback_decoder_satmae(x, y):
    x, y = callback_preprocess_satmae(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_landcover_satmae(x, y):
    x, y = callback_preprocess_landcover_satmae(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y

def callback_decoder_prithvi(x, y):
    x, y = callback_preprocess_prithvi(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y

def callback_decoder_landcover_prithvi(x, y):
    x, y = callback_preprocess_landcover_prithvi(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_geo(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder_geo(x, y)

    return x, y


def load_data(x_train, y_train, x_val, y_val, x_test, y_test, device, with_augmentations=False, num_workers=0,
              batch_size=16, downstream_task=None, model_name=None):

    """
    Loads the data from the data folder.
    """
    if model_name == 'SatMAE' or model_name == 'SatMAE_classifier':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_landcover_satmae
        else:
            cb_decoder = callback_decoder_satmae
    elif model_name == 'prithvi':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_landcover_prithvi
        else:
            cb_decoder = callback_decoder_prithvi
    else:
        if downstream_task=='lc':
            cb_decoder = callback_decoder_landcover
        elif downstream_task == 'geo':
            cb_decoder = callback_decoder_geo
        else:
            cb_decoder = callback_decoder

    if with_augmentations:
        aug = [
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                # beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
            ]

        if model_name == 'SatMAE':
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_landcover_satmae
            else:
                cb_preprocess = callback_preprocess_satmae
        
        elif model_name == 'prithvi':
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_landcover_prithvi
            else:
                cb_preprocess = callback_preprocess_prithvi
        else:
            if downstream_task=='lc':
                cb_preprocess = callback_preprocess_landcover
            else:
                cb_preprocess = callback_preprocess

        if downstream_task in ['geo', 'lc_classification', 'building_classification', 'roads_regression']:
            cb_postprocess = callback_postprocess_decoder_geo
            aug = [
                beo.AugmentationRotation(p=0.2, inplace=True),
                beo.AugmentationMirror(p=0.2, inplace=True),
                # beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
            ]
        else:
            cb_postprocess = callback_postprocess_decoder

        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=cb_preprocess,
            callback_post_augmentation=cb_postprocess,
            augmentations=aug
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=cb_decoder)

    ds_test = beo.Dataset(x_test, y_test, callback=cb_decoder)
    ds_val = beo.Dataset(x_val, y_val, callback=cb_decoder)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                          drop_last=False, generator=torch.Generator(device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                         drop_last=False, generator=torch.Generator(device))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                        drop_last=False, generator=torch.Generator(device))

    return dl_train, dl_test, dl_val

def main():
    y = np.load('/phileo_data/downstream/downstream_dataset_patches_np_HLS/east-africa_226_train_label_lc.npy')
    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)
    print()

if __name__ == '__main__':
    main()
