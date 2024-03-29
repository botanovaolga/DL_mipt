
# DATASET_PATH = "e:/Code/datasets/kitti"
DATASET_PATH = "/Users/botanovaolga/Desktop/mipt/Monocular depth/train"

# -- TRAINING CONFIGS -- #
EXPERIMENT_NAME = "BTS_Testt_new_1"  # This determines folder names used for saving tensorboard logs and model files
TOTAL_TRAIN_EPOCHS = 50

# -- TESTING CONFIGS -- #
MODEL_PATH = "models/model_latest"  # Used for testing
MAKE_VIDEO = True
VIDEO_SAVE_PATH = "video.avi"
DISPLAY_VIDEO = True