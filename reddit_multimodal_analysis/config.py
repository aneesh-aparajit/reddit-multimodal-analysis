import torch


class Config:
    seed = 101
    debug = False  # set debug=False for Full Training
    exp_name = "vit/sbert"
    model_name = "vit-sbert-multimodal"
    backbone = "google/vit-base-patch16-224+sentence-transformers/all-mpnet-base-v2"
    tokenizer = "sentence-transformers/all-mpnet-base-v2"
    image_encoder = "google/vit-base-patch16-224"
    train_bs = 24
    valid_bs = 48
    img_size = [224, 224]
    max_len = 128
    epochs = 50
    competition = "memotions-7k"

    # Optimizers
    optimizer     = 'Adam'
    learning_rate = 3e-4
    rho           = 0.9
    eps           = 1e-6
    lr_decay      = 0
    betas         = (0.9, 0.999)
    momentum      = 0
    alpha         = 0.99

    # Scheduler
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    weight_decay  = 1e-6

    # Config
    n_accumulate  = max(1, 32//train_bs)
    num_folds     = 5
    num_classes   = None

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
