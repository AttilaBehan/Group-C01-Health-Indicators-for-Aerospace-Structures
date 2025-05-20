import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import random
from time import time
import matplotlib.pyplot as plt
from Prognostic_criteria import fitness, scale_exact
from tqdm import tqdm

# Set seed for reproducibility
vae_seed = 42
random.seed(vae_seed)
tf.random.set_seed(vae_seed)
np.random.seed(vae_seed)

def VAE_merge_data(sample_filenames, target_rows=12000):
    """
    Load and flatten AE data from each sample. Interpolates each feature column to `target_rows`,
    then flattens in time-preserving order (row-major) to maintain temporal context.
    Returns a 2D array: shape = (n_samples, target_rows × 5)
    """
    rows = []
    expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
    expected_length = target_rows * len(expected_cols)

    for path in sample_filenames:
        print(f"Reading and resampling: {os.path.basename(path)}")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {path}")
        
        print("  → Columns found:", df.columns.tolist())

        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} is missing required columns: {missing}")
        df = df[expected_cols]

        df_resampled = pd.DataFrame()
        for col in df.columns:
            original = df[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, target_rows)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated

        flattened = df_resampled.values.flatten(order='C')

        print(f"  → Flattened shape: {flattened.shape[0]}")
        if flattened.shape[0] != expected_length:
            raise ValueError(
                f"ERROR: {os.path.basename(path)} vector has {flattened.shape[0]} values (expected {expected_length})"
            )

        rows.append(flattened)

    print("✅ All sample vectors have consistent shape. Proceeding to stack.")
    return np.vstack(rows)

class VAE(tf.keras.Model):
    def __init__(self, input_dim, hidden_1, hidden_2=1):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        initializer = tf.keras.initializers.GlorotUniform(seed=vae_seed)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_1, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_2 * 2, kernel_initializer=initializer, bias_initializer='zeros'),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(hidden_2,)),
            tf.keras.layers.Dense(hidden_1, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros'),
            tf.keras.layers.Dense(input_dim, kernel_initializer=initializer, bias_initializer='zeros'),
        ])

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=None):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar, z

def compute_health_indicator(x, x_recon, target_rows=12000, num_features=5):
    x_reshaped = tf.reshape(x, (-1, target_rows, num_features))
    x_recon_reshaped = tf.reshape(x_recon, (-1, target_rows, num_features))
    errors = tf.reduce_mean(tf.square(x_reshaped - x_recon_reshaped), axis=2)
    k = 1.0
    health = tf.exp(-k * errors)
    return health

def vae_loss(x, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff):
    reloss = tf.reduce_sum(tf.square(x_recon - x), axis=1)
    klloss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    diffs = health[:, 1:] - health[:, :-1]
    fealoss = tf.reduce_sum(tf.nn.relu(-diffs))
    loss = tf.reduce_mean(reloss_coeff * reloss + klloss_coeff * klloss + moloss_coeff * fealoss)
    return loss

@tf.function
def train_step(vae, batch_xs, optimizer, reloss_coeff, klloss_coeff, moloss_coeff):
    with tf.GradientTape() as tape:
        x_recon, mean, logvar, z = vae(batch_xs, training=True)
        health = compute_health_indicator(batch_xs, x_recon)
        loss = vae_loss(batch_xs, x_recon, mean, logvar, health, reloss_coeff, klloss_coeff, moloss_coeff)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

def VAE_train(sample_data, test_data, hidden_1, batch_size, learning_rate, epochs, reloss_coeff, klloss_coeff, moloss_coeff):
    n_input = sample_data.shape[1]
    hidden_2 = 1
    display = 50
    target_rows = 12000
    num_features = 5

    vae = VAE(n_input, hidden_1, hidden_2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_dataset = tf.data.Dataset.from_tensor_slices(sample_data).batch(batch_size)

    begin_time = time()
    print(f'Start training for sample data shape: {sample_data.shape}')
    for epoch in range(epochs):
        for batch_xs in train_dataset:
            loss = train_step(vae, batch_xs, optimizer, reloss_coeff, klloss_coeff, moloss_coeff)
        if epoch % display == 0:
            print(f"Epoch {epoch}, Loss = {loss.numpy()}")

    print(f"Training finished!!! Time: {time() - begin_time:.2f} seconds")

    x_recon_train, _, _, _ = vae(sample_data, training=False)
    x_recon_test, _, _, _ = vae(test_data, training=False)
    hi_train = compute_health_indicator(sample_data, x_recon_train, target_rows, num_features).numpy()
    hi_test = compute_health_indicator(test_data, x_recon_test, target_rows, num_features).numpy()

    return hi_train, hi_test

if __name__ == "__main__":
    base_path = "C:/Users/AJEBr/OneDrive/Documents/Aerospace/BsC year 2/VAE_Project/VAE_AE_DATA"
    sample_ids = [f"Sample{i}Interp.csv" for i in range(1, 13)]
    sample_paths = [os.path.join(base_path, sid) for sid in sample_ids]

    all_data = VAE_merge_data(sample_paths)
    scaler = StandardScaler().fit(all_data)
    all_data_scaled = scaler.transform(all_data).astype(np.float32)

    hidden_1 = 50
    batch_size = 5
    learning_rate = 0.0055
    epochs = 550
    reloss_coeff = 0.075
    klloss_coeff = 1.6
    moloss_coeff = 2.8

    hi_full = np.zeros((12, 1, 12000))
    fitness_scores = []
    csv_dir = base_path

    os.makedirs(csv_dir, exist_ok=True)

    for idx, sid in enumerate(tqdm(sample_ids, desc="Running VAE folds", unit="fold")):
        test_path = sample_paths[idx]
        train_paths = [p for j, p in enumerate(sample_paths) if j != idx]
        train_data = VAE_merge_data(train_paths)
        train_data_scaled = scaler.transform(train_data).astype(np.float32)

        df_test = pd.read_csv(test_path)
        if 'Time' in df_test.columns:
            df_test = df_test.drop(columns=['Time'])
        if 'Unnamed: 0' in df_test.columns:
            df_test = df_test.drop(columns=['Unnamed: 0'])
        expected_cols = ['Amplitude', 'Energy', 'Counts', 'Duration', 'RMS']
        df_test = df_test[expected_cols]
        df_resampled = pd.DataFrame()
        for col in df_test.columns:
            original = df_test[col].values
            x_original = np.linspace(0, 1, len(original))
            x_target = np.linspace(0, 1, 12000)
            interpolated = np.interp(x_target, x_original, original)
            df_resampled[col] = interpolated
        test_data = df_resampled.values.flatten(order='C').reshape(1, -1)
        test_data_scaled = scaler.transform(test_data).astype(np.float32)

        hi_train, hi_test = VAE_train(train_data_scaled, test_data_scaled, hidden_1, batch_size, learning_rate,
                                     epochs, reloss_coeff, klloss_coeff, moloss_coeff)

        train_hi_min = np.mean(hi_train, axis=1, keepdims=True)
        train_hi_max = np.max(hi_train, axis=1, keepdims=True)
        hi_test_normalized = (hi_test - train_hi_min[0:1]) / (train_hi_max[0:1] - train_hi_min[0:1] + 1e-8)
        hi_full[idx, 0, :] = hi_test_normalized[0]

        x = np.linspace(0, 100, 12000)
        fig = plt.figure()
        for i in range(hi_train.shape[0]):
            train_hi_normalized = (hi_train[i] - train_hi_min[i]) / (train_hi_max[i] - train_hi_min[i] + 1e-8)
            plt.plot(x, train_hi_normalized, color="gray", alpha=0.4, label="Train" if i == 0 else "")
        plt.plot(x, hi_test_normalized[0], color="tab:blue", linewidth=2, label=f"{sid[:-4]} (Test)")
        plt.xlabel("Lifetime (%)")
        plt.ylabel("Health Indicator")
        plt.title(f"Health Indicator - {sid[:-4]}")
        plt.legend()
        plt.savefig(os.path.join(csv_dir, f"HI_graph_{sid[:-4]}.png"))
        plt.close(fig)

        hi_train_compressed = np.array([scale_exact(hi, minimum=30) for hi in hi_train])
        ftn, mo, tr, pr, err = fitness(hi_train_compressed)
        sample_num = int(sid.replace("Sample", "").replace("Interp.csv", ""))
        fitness_scores.append([sample_num, ftn, mo, tr, pr, err])

    try:
        np.save(os.path.join(csv_dir, f"VAE_AE_seed_{vae_seed}.npy"), hi_full)
        df_fitness = pd.DataFrame(fitness_scores, columns=['Sample_ID', 'Fitness_Score', 'Monotonicity', 'Trendability', 'Prognosability', 'Error'])
        df_fitness.to_csv(os.path.join(csv_dir, 'fitness_scores.csv'), index=False)
        print(f"\n✅ All folds completed. Saved HI array to {os.path.join(csv_dir, f'VAE_AE_seed_{vae_seed}.npy')}")
        print(f"✅ Saved fitness scores to {os.path.join(csv_dir, 'fitness_scores.csv')}")
    except Exception as e:
        print(f"Error saving files: {e}")
