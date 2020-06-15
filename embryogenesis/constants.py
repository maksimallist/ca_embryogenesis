CHANNEL_N = 16  # Number of CA state channels
TARGET_PADDING = 16  # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

TARGET_EMOJI = "🦎"

EXPERIMENT_TYPE = "Regenerating"  # ["Growing", "Persistent", "Regenerating"]
EXPERIMENT_MAP = {"Growing": 0, "Persistent": 1, "Regenerating": 2}

EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

EMOJI = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'
PRETRAIN_MODELS = 'https://github.com/google-research/self-organising-systems/blob/master/assets/growing_ca/models.zip?raw=true'
# !wget -O models.zip 'https://github.com/google-research/self-organising-systems/blob/master/assets/growing_ca/models.zip?raw=true'
# !unzip -oq models.zip
