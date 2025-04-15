import os  # Added for path handling
from master import MASTERModel
import pickle
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard support

# Please install qlib first before loading the data.

universe = 'csi300'  # ['csi300','csi800']
prefix = 'opensource'  # ['original','opensource'], which training data are you using
train_data_dir = os.path.join('data', prefix)
predict_data_dir = os.path.join('data', 'opensource')

# Load training data with error handling
try:
    with open(os.path.join(train_data_dir, f"{universe}_dl_train.pkl"), "rb") as f:
        dl_train = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Training data not found in {train_data_dir}")

# Load validation and test data with error handling
try:
    with open(os.path.join(predict_data_dir, f"{universe}_dl_valid.pkl"), "rb") as f:
        dl_valid = pickle.load(f)
    with open(os.path.join(predict_data_dir, f"{universe}_dl_test.pkl"), "rb") as f:
        dl_test = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Prediction data not found in {predict_data_dir}")

print("Data Loaded.")

d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 5
elif universe == 'csi800':
    beta = 2

n_epoch = 200  # Updated to train for 10 epochs
lr = 1e-5
GPU = 0  # Force CPU usage
train_stop_loss_thred = 0.95

ic = []
icir = []
ric = []
ricir = []

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join('logs', f'{universe}_{prefix}'))

# Training
######################################################################################
enable_training = True  # Set to True to enable training
if enable_training:
    for seed in [0]: #[0, 1, 2, 3, 4]
        model = MASTERModel(
            d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, train_stop_loss_thred=train_stop_loss_thred,
            save_path='model', save_prefix=f'{universe}_{prefix}'
        )

        start = time.time()
        # Train
        model.fit(dl_train, dl_valid)  # Ensure the model is trained before calling predict
        print("Model Trained.")

        # Test
        predictions, metrics = model.predict(dl_test)
        writer.add_scalar(f'Test/IC_seed_{seed}', metrics['IC'], 0)
        writer.add_scalar(f'Test/ICIR_seed_{seed}', metrics['ICIR'], 0)
        writer.add_scalar(f'Test/RIC_seed_{seed}', metrics['RIC'], 0)
        writer.add_scalar(f'Test/RICIR_seed_{seed}', metrics['RICIR'], 0)

        running_time = time.time() - start

        print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
        print(metrics)

        ic.append(metrics['IC'])
        icir.append(metrics['ICIR'])
        ric.append(metrics['RIC'])
        ricir.append(metrics['RICIR'])
######################################################################################

# Close TensorBoard writer
writer.close()

# Load and Test
######################################################################################
for seed in [0]:
    param_path = os.path.join('model', f"{universe}_{prefix}_{seed}.pkl")

    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Model parameter file not found: {param_path}")

    print(f'Model Loaded from {param_path}')
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, train_stop_loss_thred=train_stop_loss_thred,
        save_path='model/', save_prefix=universe
    )
    model.load_param(param_path)  # Ensure CPU compatibility
    predictions, metrics = model.predict(dl_test)
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
######################################################################################

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))