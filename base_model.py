import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', writer=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device("cpu")  # Force CPU usage
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            # Removed torch.cuda.manual_seed_all(self.seed) since GPU is not available
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix
        self.writer = writer  # TensorBoard SummaryWriter


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        batch_idx = 0

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label) # CSZscoreNorm
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            # Log batch-level metrics if writer is available
            if self.writer is not None and self.fitted is not None:
                # Only log every 10 batches to avoid overwhelming TensorBoard
                if batch_idx % 10 == 0:
                    global_step = self.fitted * len(data_loader) + batch_idx
                    self.writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
                    
                    # Log feature statistics
                    if batch_idx == 0:  # Only log once per epoch to save space
                        self.writer.add_histogram(f'Epoch_{self.fitted}/FeatureValues', feature.float(), global_step)
                        self.writer.add_histogram(f'Epoch_{self.fitted}/Labels', label, global_step)
                        self.writer.add_histogram(f'Epoch_{self.fitted}/Predictions', pred, global_step)
            
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
            
            batch_idx += 1

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test. 
            label = zscore(label)
                        
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))  # Ensure CPU compatibility
        self.fitted = 0  # Set to 0 to indicate the model is loaded and ready for prediction

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        
        # Log model graph to TensorBoard if writer is available
        if self.writer is not None:
            # Get a sample batch to trace the model
            for data in train_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1].to(self.device).float()
                self.writer.add_graph(self.model, feature)
                break
        
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step  # Update fitted to indicate the model is trained
            
            # Log training loss to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Training/Loss', train_loss, step)
                
                # Log model parameters histograms
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.writer.add_histogram(f"Parameters/{name}", param.data, step)
                        if param.grad is not None:
                            self.writer.add_histogram(f"Gradients/{name}", param.grad, step)
            
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
                
                # Log validation metrics to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Validation/IC', metrics['IC'], step)
                    self.writer.add_scalar('Validation/ICIR', metrics['ICIR'], step)
                    self.writer.add_scalar('Validation/RIC', metrics['RIC'], step)
                    self.writer.add_scalar('Validation/RICIR', metrics['RICIR'], step)
            else: 
                print("Epoch %d, train_loss %.6f" % (step, train_loss))
        
            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                
                # Log model at best epoch
                if self.writer is not None:
                    self.writer.add_text('Training/BestModel', f'Best model saved at epoch {step} with loss {train_loss:.6f}', step)
                break

        # Save the best model parameters if training completes all epochs
        if best_param is None:
            best_param = copy.deepcopy(self.model.state_dict())
            torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')

    def predict(self, dl_test):
        if self.fitted < 0:  # Ensure this comparison works correctly
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            
            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return predictions, metrics
