import numpy as np
from sklearn.model_selection import train_test_split

from utils import get_physionet_data
from models.ecg import ECGModel


# Load Physionet ECG data
_, data = get_physionet_data()
print('%i subjects loaded' % len(data))

# Train-validation split
ids = np.array(list(data.keys()))
train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=38)

train_data = [data[id] for id in train_ids]
val_data = [data[id] for id in val_ids]

# Train and save weights
trained_model_path = './trained_models/physionet'
model = ECGModel(latent_space_dim=8, batch_size=64, hidden_dim=128, learning_rate=5e-4, log_dir=trained_model_path)
model.fit(train_data, val_data)

model.save_weights(trained_model_path)
