import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 1. Define your VAE model (updated for dynamic sequence lengths)
class RegularizedHealthVAE(Model):
    def __init__(self, num_features=201, latent_dim=32, 
                 dropout_rate=0.3, l2_weight=1e-4, lstm_units=64):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        
        # L2 regularization
        kernel_reg = regularizers.l2(l2_weight)
        
        # Feature weights
        self.feature_dropout = layers.Dropout(dropout_rate)
        self.feature_weights = tf.Variable(
            tf.ones(num_features),
            trainable=True,
            constraint=lambda x: tf.nn.softmax(x)
        )

        # Encoder (now handles variable length sequences)
        self.encoder = tf.keras.Sequential([
            layers.Masking(mask_value=0.),  # Handles padded sequences
            layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            layers.Dense(256, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            layers.Dense(2 * latent_dim)  # mean and logvar
        ])
        
        # Decoder (generates sequences of arbitrary length)
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            layers.RepeatVector(1),  # Will be dynamically adjusted in call()
            layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=kernel_reg),
            layers.Dropout(dropout_rate),
            layers.TimeDistributed(layers.Dense(num_features, kernel_regularizer=kernel_reg))
        ])
        
        # Health tracking
        self.healthy_ref = tf.Variable(tf.zeros(latent_dim), trainable=False)
        self.alpha = tf.Variable(0.7, trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, 1))

    def call(self, x, seq_len=None, training=False):
        # Dynamic sequence length handling
        if seq_len is None:
            seq_len = tf.shape(x)[1]
        
        # Feature dropout
        if training:
            x = self.feature_dropout(x)
        
        # Encoding
        mean, logvar = tf.split(self.encoder(x, training=training), 2, axis=1)
        z = self.reparameterize(mean, logvar)
        
        # Dynamic decoding
        self.decoder.layers[2] = layers.RepeatVector(seq_len)  # Update sequence length
        x_recon = self.decoder(z, training=training)
        
        return x_recon, mean, logvar, z
    
    # [Keep all other methods unchanged...]

# 2. Batch Generator
class HealthDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=4, shuffle=True):
        """
        Args:
            data: Processed numpy array (n_samples, max_seq_len, n_features + 1)
            batch_size: Number of samples per batch
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = data.shape[0]
        self.max_seq_len = data.shape[1]
        self.indices = np.arange(self.n_samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = self.data[batch_indices]
        
        # Extract features and times
        X = batch_data[..., 1:]  # Features
        t = batch_data[..., 0]   # Time
        
        # Create sequence masks (for variable length if needed)
        masks = tf.cast(t != 0, tf.float32)  # Assuming 0-padding
        
        return (X, t, masks), None  # None for dummy labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# 3. Data Preparation
def prepare_data(processed_data, test_size=0.2):
    """Split into train/val and create generators"""
    # Split samples (not timesteps)
    train_data, val_data = train_test_split(
        processed_data, 
        test_size=test_size,
        shuffle=True,
        random_state=42
    )
    
    return train_data, val_data

# 4. Training Pipeline
def train_vae(processed_data):
    # Data preparation
    train_data, val_data = prepare_data(processed_data)
    
    # Create generators
    train_gen = HealthDataGenerator(train_data, batch_size=4)
    val_gen = HealthDataGenerator(val_data, batch_size=4, shuffle=False)
    
    # Model initialization
    n_features = processed_data.shape[2] - 1  # Subtract time column
    vae = RegularizedHealthVAE(
        num_features=n_features,
        latent_dim=32,
        dropout_rate=0.4,
        l2_weight=1e-4,
        lstm_units=64
    )
    
    # Custom training step (handles masks)
    @tf.function
    def train_step(inputs, _):
        x, t, masks = inputs
        with tf.GradientTape() as tape:
            x_recon, mean, logvar, z = vae(x, seq_len=tf.shape(x)[1], training=True)
            
            # Masked losses
            recon_loss = tf.reduce_sum(masks * tf.square(x - x_recon)) / tf.reduce_sum(masks)
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
            
            health = vae.compute_health_indicator(x, x_recon, mean, logvar)
            mono_loss = vae.monotonicity_loss(health)
            trend_loss = vae.trendability_loss(health, t)
            
            total_loss = 0.5*recon_loss + 0.3*kl_loss + 0.1*mono_loss + 0.1*trend_loss
        
        gradients = tape.gradient(total_loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return total_loss

    # Compile with dummy optimizer
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    # Training loop
    history = {'loss': [], 'val_loss': []}
    for epoch in range(300):
        # Training
        epoch_loss = []
        for batch in train_gen:
            loss = train_step(*batch)
            epoch_loss.append(loss.numpy())
        
        # Validation
        val_loss = []
        for val_batch in val_gen:
            x_val, t_val, masks_val = val_batch[0]
            x_recon, mean, logvar, z = vae(x_val, training=False)
            loss = tf.reduce_mean(tf.square(x_val - x_recon))  # Simple recon loss for validation
            val_loss.append(loss.numpy())
        
        history['loss'].append(np.mean(epoch_loss))
        history['val_loss'].append(np.mean(val_loss))
        
        print(f"Epoch {epoch+1}: Loss {history['loss'][-1]:.4f}, Val Loss {history['val_loss'][-1]:.4f}")
    
    return vae, history

# 5. Run the training
vae_model, training_history = train_vae(processed_data)