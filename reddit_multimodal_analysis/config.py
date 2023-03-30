import torch


class Config:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = 'vit/sbert'
    model_name    = 'vit-sbert-multimodal'
    backbone      = 'google/vit-base-patch16-224+sentence-transformers/all-mpnet-base-v2'
    tokenizer     = 'sentence-transformers/all-mpnet-base-v2'
    image_encoder = 'google/vit-base-patch16-224'
    train_bs      = 24
    valid_bs      = 48
    img_size      = [224, 224]
    max_len       = 128
    epochs        = 50
    lr            = 5e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(100*6*1.8)
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = 32//train_bs
    n_fold        = 5
    num_classes   = 3
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    competition   = 'memotions-7k'