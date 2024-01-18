import torch
import gc

import training_script
from utils.training_utils import read_yaml

torch.cuda.set_device(3)

parser, parser_yaml = training_script.get_args()
args_yaml, remainder = parser_yaml.parse_known_args()

if args_yaml.read_yaml is not None:
    print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
    args = read_yaml(args_yaml.read_yaml)
else:
    args = parser.parse_args()


for downstream_task in ['building', 'lc']: #  

    args['downstream_task'] = downstream_task
    if downstream_task == 'lc':
        args['output_channels'] = 11
        metric = 'acc'
    else:
        args['output_channels'] = 1
        metric = 'msc'

    experiment_name = args['experiment_name']
    folder = f'/home/phimultigpu/phileo_NFS/phileo_data/experiments/{experiment_name}/{downstream_task}/'

    for split_ratio in [1.0, ]:  
        model_list = ['vit_cnn_base', 'vit_cnn', 'vit_cnn_gc', 'vit_cnn_gc_wSkip'] 

        # 'GeoAware_core_nano', 'vit_cnn_gc_wSkip' 'GeoAware_contrastive_core_nano', 'vit_cnn_base', 'GeoAware_mh_pred_core_nano', 'core_unet_nano', 'SatMAE', 'seasonal_contrast', 'resnet_imagenet', 'prithvi', 'vit_cnn', 'vit_cnn_gc'
        # 'vit_cnn_base', 'vit_cnn', 'vit_cnn_gc'
        

        for model_name in model_list:
            args['split_ratio'] = split_ratio
            args['model_name'] = model_name
            args['pretrained_model_path'] = None
            args['pretrained_model_path'] = None
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            gc.collect()


            if model_name == 'GeoAware_contrastive_core_nano':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt'
            elif model_name == 'GeoAware_core_nano':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_8.pt'
            elif model_name == 'GeoAware_mh_pred_core_nano':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/01112023_CoreEncoderMultiHead_geo_pred_geo_reduce_on_plateau/CoreEncoderMultiHead_geo_pred_best.pt'
            elif model_name == 'prithvi':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/Prithvi_100M.pt'
            elif model_name == 'SatMAE':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/SatMAE_pretrain-vit-large-e199.pth'
                
            elif model_name == 'vit_cnn' or model_name == 'vit_cnn_wSkip':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/31102023_MaskedAutoencoderViT/MaskedAutoencoderViT_ckpt.pt'
            elif model_name == 'vit_cnn_gc' or model_name == 'vit_cnn_gc_wSkip':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/03112023_MaskedAutoencoderGroupChannelViT/MaskedAutoencoderGroupChannelViT_ckpt.pt'
            elif model_name == 'seasonal_contrast':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/seco_resnet50_1m.ckpt'
                

            if model_name == 'core_unet_nano' or model_name == 'vit_cnn_base' or model_name == 'resnet_imagenet':
                args['pretrained_model_path'] = None
                training_script.main(**vars(args))
          
            else:
                for freeze in [False, True]:
                    args['freeze_pretrained'] = freeze
                    training_script.main(**vars(args))


