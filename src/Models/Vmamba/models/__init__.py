import os
from functools import partial
import torch

from .vmamba import VSSM


def build_vssm_model(config, **kwargs):
    model_type = config.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.PATCH_SIZE, 
            in_chans=config.IN_CHANS, 
            num_classes=config.NUM_CLASSES, 
            depths=config.DEPTHS, 
            dims=config.EMBED_DIM, 
            # ===================
            ssm_d_state=config.SSM_D_STATE,
            ssm_ratio=config.SSM_RATIO,
            ssm_rank_ratio=config.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.SSM_DT_RANK == "auto" else int(config.SSM_DT_RANK)),
            ssm_act_layer=config.SSM_ACT_LAYER,
            ssm_conv=config.SSM_CONV,
            ssm_conv_bias=config.SSM_CONV_BIAS,
            ssm_drop_rate=config.SSM_DROP_RATE,
            ssm_init=config.SSM_INIT,
            forward_type=config.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MLP_RATIO,
            mlp_act_layer=config.MLP_ACT_LAYER,
            mlp_drop_rate=config.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.DROP_PATH_RATE,
            patch_norm=config.PATCH_NORM,
            norm_layer=config.NORM_LAYER,
            downsample_version=config.DOWNSAMPLE,
            patchembed_version=config.PATCHEMBED,
            gmlp=config.GMLP,
            use_checkpoint=config.USE_CHECKPOINT,
            # ===================
            posembed=config.POSEMBED,
            imgsize=config.IMG_SIZE,
        )
        return model

    return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config)
    if model is None:
        from .simvmamba import simple_build
        model = simple_build(config.TYPE)
    return model




