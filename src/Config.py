import os

class config:
    # 根目录
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, '../dataset/PestDetectionv5Dataset/csv')
    train_data_path = os.path.join(root_path, 'train/train_data.csv')
    test_data_path = os.path.join(root_path, 'test/test_data.csv')
    output_path = os.path.join(root_path, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    # 一般超参
    epoch = 20
    learning_rate = 3e-5
    weight_decay = 0
    num_labels = 4
    loss_weight = [1.68, 9.3, 3.36]

    # Fuse相关
    fuse_model_type = 'NaiveCombine'
    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128

    # BERT相关
    fixed_text_model_params = False
    bert_name = 'roberta-base'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2

    # ResNet相关
    resnet_name = 'resnet50d'
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64


    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 4, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': 4, 'shuffle': False, 'num_workers': 0}
    test_params =  {'batch_size': 4, 'shuffle': False, 'num_workers': 0}
    
    #VMamba
    TYPE = 'vssm'
    PATCH_SIZE = 4
    IN_CHANS = 3
    NUM_CLASSES = num_labels
    DEPTHS = [2, 2, 9, 2]
    EMBED_DIM = 96
    SSM_D_STATE = 16
    SSM_RATIO = 2.0
    SSM_RANK_RATIO = 2.0
    SSM_DT_RANK = "auto"
    SSM_ACT_LAYER = "silu"
    SSM_CONV = 3
    SSM_CONV_BIAS = True
    SSM_DROP_RATE = 0.0
    SSM_INIT = "v0"
    SSM_FORWARDTYPE = "v2"
    MLP_RATIO = 4.0
    MLP_ACT_LAYER = "gelu"
    MLP_DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    PATCH_NORM = True
    NORM_LAYER = "ln"
    DOWNSAMPLE = "v2"
    PATCHEMBED = "v2"
    GMLP = False
    USE_CHECKPOINT = False
    POSEMBED = False
    IMG_SIZE = 224

    
    