""" DCEC model aiming at damage mode classification in ReMAP sample data. """

# Imports
import os
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Run file on GPU (or CPU if necessary)
device = "cuda"
torch.device(device)
print(f"Using device: {device}")

# Sets all random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class ReMAP_Dataset(Dataset):
    """ Load and preprocesses the ReMAP dataset. """
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['Amplitude', 'Rise-Time', 'Energy', 'Counts', 'Duration', 'RMS']]  # Low-level features which are used
    X_scaled = StandardScaler().fit_transform(X)  # Maybe apply adaptive standardization?
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    return X_tensor.to(device)


class CAE(nn.Module):
    """ Generates a 1D convolutional autoencoder (CAE). """

    def __init__(self, input_size, latent_dim=10):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_size, latent_dim))
        self.decoder_fc = nn.Linear(latent_dim, 32 * input_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, input_size)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1))

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(self.decoder_fc(z))
        return z, x_recon

class ClusteringLayer(nn.Module):
    """ Generates the clustering layer. """

    def __init__(self, n_clusters, latent_dim):
        super(ClusteringLayer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2))
        q = q.pow((1 + 1) / 2)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


def target_distribution(q):
    """ Student's t-distribution. """

    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


def train_dcec(X_tensor, input_size, n_clusters=5, latent_dim=10, batch_size=256, pretrain_epochs=5, joint_epochs=30, learning_rate=1e-3):
    """ Trains the DCEC model. """
    # Hyperparameters might need some tweaking

    dataset = ReMAP_Dataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder = CAE(input_size, latent_dim).to(device)
    clustering = ClusteringLayer(n_clusters, latent_dim).to(device)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(clustering.parameters()), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Pretraining
    for epoch in range(pretrain_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z, x_recon = autoencoder(batch)
            loss = loss_fn(x_recon, batch)
            loss.backward()
            optimizer.step()

    # Initialize cluster centers
    autoencoder.eval()
    all_z = []
    with torch.no_grad():
        for batch in dataloader:
            z, _ = autoencoder(batch)
            all_z.append(z)
    all_z = torch.cat(all_z).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20).fit(all_z)
    clustering.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # Joint training
    autoencoder.train()
    for epoch in range(joint_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z, x_recon = autoencoder(batch)
            q = clustering(z)
            p = target_distribution(q).detach()
            loss_recon = loss_fn(x_recon, batch)
            loss_kl = nnf.kl_div(q.log(), p, reduction='batchmean')
            loss = loss_recon + 0.1 * loss_kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{joint_epochs} - Total Loss: {total_loss:.4f}")

    return autoencoder, clustering


def infer_clusters(autoencoder, clustering, X_tensor):
    """ Inference for clusters. """

    autoencoder.eval()
    clustering.eval()
    with torch.no_grad():
        z, _ = autoencoder(X_tensor.to(device))
        q = clustering(z)
        labels = torch.argmax(q, dim=1).cpu().numpy()
    return labels


def loocv_run():
    """ Runs leave-one-out cross-validation (LOOCV): 10 training samples, 1 validation sample and 1 test sample.
        Each sample is the test sample once. """

    all_indices = list(range(1, 13))
    n_clusters = 5
    latent_dim = 10

    total_runs = len(all_indices)
    start_time = time.time()

    for i in tqdm(range(total_runs), desc="LOOCV Progress"):
        loop_start = time.time()

        val_idx = all_indices[i]
        test_idx = all_indices[(i + 1) % total_runs]  # Wrap around at the end
        train_indices = [j for j in all_indices if j not in (val_idx, test_idx)]

        print(f"\n=== Training on Samples {train_indices}, Validation on Sample {val_idx}, Test on Sample {test_idx} ===")

        # Load and concatenate training data
        train_tensors = [load_data(f"ReMAP_Data/Sample{j}.csv") for j in train_indices]
        X_train = torch.cat(train_tensors)
        input_size = X_train.size(2)

        autoencoder, clustering = train_dcec(X_train, input_size, n_clusters, latent_dim)

        # Save autoencoder and clustering models
        model_dir = f"DCEC_Models/Train_{'_'.join(map(str, train_indices))}"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(autoencoder.state_dict(), os.path.join(model_dir, "autoencoder.pth"))
        torch.save(clustering.state_dict(), os.path.join(model_dir, "clustering.pth"))

        # Save output for each training sample separately
        for train_sample_idx in train_indices:
            X_train_sample = load_data(f"ReMAP_Data/Sample{train_sample_idx}.csv")
            labels_train_sample = infer_clusters(autoencoder, clustering, X_train_sample)
            df_train_sample = pd.read_csv(f"ReMAP_Data/Sample{train_sample_idx}.csv")
            df_train_sample["Cluster"] = labels_train_sample
            df_train_sample.to_csv(
                f"DCEC_Training_Output/Train_Sample{train_sample_idx}_Val{val_idx}_Test{test_idx}.csv",
                index=False)

        # Predict on validation sample
        X_val = load_data(f"ReMAP_Data/Sample{val_idx}.csv")
        labels_val = infer_clusters(autoencoder, clustering, X_val)
        df_val = pd.read_csv(f"ReMAP_Data/Sample{val_idx}.csv")
        df_val["Cluster"] = labels_val
        df_val.to_csv(f"DCEC_Validation_Output/Validation_Sample{val_idx}_Test{test_idx}.csv", index=False)

        # Predict on test sample
        X_test = load_data(f"ReMAP_Data/Sample{test_idx}.csv")
        labels_test = infer_clusters(autoencoder, clustering, X_test)
        df_test = pd.read_csv(f"ReMAP_Data/Sample{test_idx}.csv")
        df_test["Cluster"] = labels_test
        df_test.to_csv(f"DCEC_Testing_Output/Test_Sample{test_idx}_Val{val_idx}.csv", index=False)

        # Estimate remaining time
        loop_end = time.time()
        elapsed = loop_end - loop_start
        remaining = (total_runs - i - 1) * elapsed
        print(f"Iteration time: {elapsed:.1f}s, Estimated time remaining: {remaining/60:.1f} minutes")

    total_time = time.time() - start_time
    print(f"\nLOOCV completed in {total_time/60:.2f} minutes.")

loocv_run()
