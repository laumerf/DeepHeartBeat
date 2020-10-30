import numpy as np
from sklearn.model_selection import KFold

from models.echo import EchocardioModel
from utils import *


# Load EchoNet-Dynamic data
data_info, files = load_data_echonet()

# Only use train and val splits
echonet_train_ids = data_info[data_info.Split == 'TRAIN'].index.values
echonet_val_ids = data_info[data_info.Split == 'VAL'].index.values
ids = list(echonet_train_ids) + list(echonet_val_ids)

# 5-fold CV training -> 5 models
files = np.array([files[id] for id in ids])
kf = KFold(n_splits=5, shuffle=True, random_state=230)
for i, (train_index, val_index) in enumerate(kf.split(files)):

    train_files = files[train_index]
    val_files = files[val_index]

    trained_model_path = './trained_models/echonet_dynamic_' + str(i)

    model = EchocardioModel(latent_space_dim=128, batch_size=32, hidden_dim=128, log_dir=trained_model_path)
    model.fit(train_files, val_files)

    model.save_weights(trained_model_path)