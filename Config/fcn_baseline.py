"""
FCN baseline experiment configuration.

Model:
    FCN-8s

Backbone:
    VGG16

Dataset:
    Semantic Segmentation Dataset

Purpose:
    Baseline experiment
"""


class Config:

    # =====================================================
    # Experiment Information
    # =====================================================

    # Experiment name
    # Used as:
    #
    # Experiments/{EXP_NAME}

    EXP_NAME = "fcn_baseline"
    DESCRIPTION = (
        "FCN-8s with VGG16 backbone baseline experiment"
    )

    # =====================================================
    # Environment
    # =====================================================

    DEVICE = "cuda"
    NUM_GPUS = 1
    SEED = 42

    # =====================================================
    # Dataset Configuration
    # =====================================================
    # Kaggle dataset mount path.
    # When you upload a Dataset named "pascal-voc-2012" on Kaggle,
    # it is mounted at /kaggle/input/pascal-voc-2012/.
    # Change this path according to your actual Kaggle Dataset name.
    DATA_ROOT = "/kaggle/input/pascal-voc-2012/VOC2012"

    # Dataset structure:
    #
    # dataset/
    #
    #   train/
    #       images/
    #       masks/
    #
    #   val/
    #       images/
    #       masks/

    TRAIN_DIR = "train"
    VAL_DIR = "val"
    IMAGE_DIR = "images"
    MASK_DIR = "masks"

    # Number of categories
    NUM_CLASSES = 21

    # Ignore label
    IGNORE_INDEX = 255

    # =====================================================
    # Data Processing
    # =====================================================
    IMAGE_SIZE = (
        512,
        512
    )
    NORMALIZE = True

    # ImageNet normalization

    MEAN = [0.485,0.456,0.406]
    STD = [0.229,0.224,0.225]

    # =====================================================
    # Data Augmentation
    # =====================================================
    USE_AUGMENTATION = True
    RANDOM_HORIZONTAL_FLIP = 0.5
    RANDOM_ROTATION = 10
    RANDOM_CROP = False
    COLOR_JITTER = True
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1

    # =====================================================
    # DataLoader
    # =====================================================
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DROP_LAST = False

    # =====================================================
    # Model Configuration
    # =====================================================
    MODEL_NAME = "FCN"
    MODEL_TYPE = "FCN8s"

    # -----------------
    # Backbone
    # -----------------

    BACKBONE = "VGG16"
    PRETRAINED = True
    PRETRAINED_PATH = None

    # Freeze backbone
    FREEZE_BACKBONE = False

    # =====================================================
    # FCN Specific Parameters
    # =====================================================
    # FCN skip connection
    USE_SKIP_CONNECTION = True

    # FCN upsampling method
    UPSAMPLE_MODE = "bilinear"
    ALIGN_CORNERS = False

    # =====================================================
    # Loss Configuration
    # =====================================================

    LOSS_TYPE = "CrossEntropy"

    # Cross entropy weight
    USE_CLASS_WEIGHT = False
    CLASS_WEIGHT = None

    # Dice Loss
    USE_DICE_LOSS = False
    DICE_WEIGHT = 1.0

    # Focal Loss
    USE_FOCAL_LOSS = False
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2

    # =====================================================
    # Optimizer
    # =====================================================
    OPTIMIZER = "Adam"
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    BETAS = (
        0.9,
        0.999
    )
    MOMENTUM = 0.9

    # =====================================================
    # Learning Rate Scheduler
    # =====================================================
    USE_SCHEDULER = True
    SCHEDULER = "CosineAnnealing"
    T_MAX = 100
    MIN_LR = 1e-6

    # =====================================================
    # Training Configuration
    # =====================================================
    EPOCHS = 100
    START_EPOCH = 0
    ACCUMULATION_STEPS = 1
    AMP = True
    # Gradient clipping
    GRAD_CLIP = None

    # =====================================================
    # Validation
    # =====================================================
    VAL_INTERVAL = 1
    SAVE_BEST_ONLY = True

    # Metrics
    METRICS = [
        "pixel_accuracy",
        "mIoU",
        "Dice"
    ]

    # =====================================================
    # Checkpoint
    # =====================================================
    SAVE_CHECKPOINT = True
    CHECKPOINT_INTERVAL = 10
    RESUME = False
    RESUME_PATH = None

    # =====================================================
    # Experiment Output
    # =====================================================
    SAVE_DIR = "Experiments"
    SAVE_LOG = True
    SAVE_PREDICTION = True
    PREDICTION_DIR = "pred_images"
    SAVE_CONFIG = True

    # =====================================================
    # Debug
    # =====================================================
    DEBUG = False
    PRINT_FREQ = 20