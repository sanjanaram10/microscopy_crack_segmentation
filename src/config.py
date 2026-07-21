import torch

class Config:
    # Paths
    DATA_DIR = "dataset/processed"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "outputs"

    # Model
    NUM_CLASSES = 2  # background + crack
    BACKBONE = "resnet50"
    PRETRAINED = True

    # Training
    DEVICE = "cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 40
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    LR_STEP_SIZE = 15
    LR_GAMMA = 0.5
    GRAD_ACCUM_STEPS = 2

    # Image
    IMG_SIZE = 512
    PIXEL_SIZE_MM = 0.05  # calibration factor

    # Inference
    CONFIDENCE_THRESHOLD = 0.3
    MIN_INSTANCE_AREA_PX = 50