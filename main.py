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
train_stop_loss_thred = 0.55

ic = []
icir = []
ric = []
ricir = []

# Initialize TensorBoard writer with more detailed log directory
log_dir = os.path.join('logs', f'{universe}_{prefix}_{time.strftime("%Y%m%d-%H%M%S")}')
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logs will be saved to {log_dir}")

# Function to log model architecture as text
def log_model_architecture(writer):
    # Create a text description of the MASTER model architecture
    architecture = """
    MASTER Model Architecture:
    
    1. Input Layer:
       - Feature dimension: {d_feat}
       - Model dimension: {d_model}
    
    2. Feature Gate:
       - Gate input dimension: {gate_input_dim}
       - Beta temperature: {beta}
    
    3. Temporal Attention (Intra-stock):
       - Number of heads: {t_nhead}
       - Dropout rate: {dropout}
       - Processes time-series data for each stock
    
    4. Spatial Attention (Inter-stock):
       - Number of heads: {s_nhead}
       - Dropout rate: {dropout}
       - Captures relationships between different stocks
    
    5. Temporal Aggregation:
       - Aggregates temporal information
    
    6. Output Layer:
       - Predicts stock returns
    """.format(
        d_feat=d_feat,
        d_model=d_model,
        gate_input_dim=gate_input_end_index-gate_input_start_index,
        beta=beta,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        dropout=dropout
    )
    
    writer.add_text('Model/Architecture', architecture, 0)

# Add model hyperparameters to TensorBoard
writer.add_text('Hyperparameters/Model', 
                f"d_feat: {d_feat}, d_model: {d_model}, t_nhead: {t_nhead}, s_nhead: {s_nhead}, " +
                f"dropout: {dropout}, beta: {beta}, n_epoch: {n_epoch}, lr: {lr}", 0)
writer.add_text('Hyperparameters/Data', 
                f"universe: {universe}, prefix: {prefix}, " +
                f"gate_input_start_index: {gate_input_start_index}, gate_input_end_index: {gate_input_end_index}", 0)

# Log model architecture
log_model_architecture(writer)

# Training
######################################################################################
enable_training = True  # Set to True to enable training
if enable_training:
    for seed in [0]: #[0, 1, 2, 3, 4]
        model = MASTERModel(
            d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, train_stop_loss_thred=train_stop_loss_thred,
            save_path='model', save_prefix=f'{universe}_{prefix}', writer=writer
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
    # Create a new writer for test results
    test_writer = SummaryWriter(log_dir=os.path.join('logs', f'{universe}_{prefix}_test_{time.strftime("%Y%m%d-%H%M%S")}'))
    print(f"Test TensorBoard logs will be saved to {test_writer.log_dir}")
    
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr=lr, GPU=GPU, seed=seed, train_stop_loss_thred=train_stop_loss_thred,
        save_path='model/', save_prefix=universe, writer=test_writer
    )
    model.load_param(param_path)  # Ensure CPU compatibility
    predictions, metrics = model.predict(dl_test)
    print(metrics)
    
    # Log test metrics to TensorBoard
    test_writer.add_scalar(f'Test/IC_seed_{seed}', metrics['IC'], 0)
    test_writer.add_scalar(f'Test/ICIR_seed_{seed}', metrics['ICIR'], 0)
    test_writer.add_scalar(f'Test/RIC_seed_{seed}', metrics['RIC'], 0)
    test_writer.add_scalar(f'Test/RICIR_seed_{seed}', metrics['RICIR'], 0)
    
    # Log model architecture for the test run
    log_model_architecture(test_writer)
    
    # Close the test writer
    test_writer.close()

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
######################################################################################

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))
