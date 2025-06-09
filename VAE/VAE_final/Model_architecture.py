import tensorflow as tf
import numpy as np

class VAE_Seed():
    vae_seed = 42

# Defines Keras VAE model
class VAE(tf.keras.Model):
    # Contructor method which initializes VAE, hidden_2 = size of latent space, usually smaller than hidden_1
    def __init__(self, timesteps_per_batch, n_features, hidden_1, hidden_2, dropout_rate=0.2):
        # Calls parent class constructor to initialize model properly
        super(VAE, self).__init__()

        # Storing model parameters
        self.timesteps = int(timesteps_per_batch)
        self.n_features = int(n_features)
        # Handle NumPy integers explicitly
        self.hidden_1 = int(hidden_1.item()) if isinstance(hidden_1, np.integer) else int(hidden_1)
        self.hidden_2 = int(hidden_2.item()) if isinstance(hidden_2, np.integer) else int(hidden_2)
        hidden_1_div_2 = int(self.hidden_1 // 2)
        print(f"After conversion: hidden_1={self.hidden_1} ({type(self.hidden_1)}), "
              f"hidden_2={self.hidden_2} ({type(self.hidden_2)}), "
              f"hidden_1_div_2={hidden_1_div_2} ({type(hidden_1_div_2)}), "
              f"timesteps={self.timesteps} ({type(self.timesteps)}), "
              f"n_features={self.n_features} ({type(self.n_features)})")
        # Check types after conversion
        if not isinstance(self.hidden_1, int):
            raise ValueError(f"self.hidden_1 ({self.hidden_1}) is not a Python int")
        if not isinstance(self.hidden_2, int):
            raise ValueError(f"self.hidden_2 ({self.hidden_2}) is not a Python int")
        if not isinstance(hidden_1_div_2, int):
            raise ValueError(f"hidden_1//2 ({hidden_1_div_2}) is not a Python int")
        if not isinstance(self.timesteps, int):
            raise ValueError(f"timesteps_per_batch ({self.timesteps}) is not a Python int")
        if not isinstance(self.n_features, int):
            raise ValueError(f"n_features ({self.n_features}) is not a Python int")
        #if not isinstance(hidden_1, (int)):
         #   raise ValueError(f"hidden_1 ({hidden_1}) is not an integer")
        #hidden_1_div_2 = int(self.hidden_1 // 2)
        # Initialization of weights (to improve stability of training, with seed for reproducability)
        initializer = tf.keras.initializers.GlorotUniform(seed=VAE_Seed.vae_seed)

        # Encoder Network 
            # Sequential = linear stack of layers
            # layers: input (with input dim), dense (hidden_1 with signoid activation function), dense (hidden_2 * 2, bc outputs mean and log-variance)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(timesteps_per_batch, n_features)),
            # LSTM processes sequences and returns last output
            tf.keras.layers.LSTM(hidden_1, activation='tanh', kernel_initializer=initializer, return_sequences=True),
            tf.keras.layers.Dropout(dropout_rate),  # <-- HERE
            tf.keras.layers.LSTM(hidden_1_div_2, activation='tanh'),  # Reduced dim
            # Output mean and log-variance of latent space
            tf.keras.layers.Dense(hidden_2 * 2, kernel_initializer=initializer),
        ])


        # Decoder Network
            # Takes latent space (hidden_2) as input, then reconstructs by reversing Encoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(hidden_2,)),
            # Project to intermediate dimension
            tf.keras.layers.Dense(hidden_1 * 2, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            # Project back to original LSTM dim
            tf.keras.layers.Dense(hidden_1),
            # Sequence reconstruction
            tf.keras.layers.RepeatVector(timesteps_per_batch),
            # Stacked LSTMs for better sequence modeling
            tf.keras.layers.LSTM(hidden_1, return_sequences=True),
            tf.keras.layers.LSTM(n_features, return_sequences=True),
            # Properly scaled output
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(n_features, activation='linear')),
        ])

    # Encoding method
    def encode(self, x):
        mean_logvar = self.encoder(x)  # Passes input 'x' through encoder
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)  # Splits outputs mean and log(var) 
        return mean, logvar

    # Reparametrization trick 
        # Enables backpropagation through random sampling
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))  # Samples noise from standard normal distribution
        return mean + tf.exp(0.5 * logvar) * eps  # calculates z = mu + sigma * epsilon, where sigma = exp(0.5*logvar)

    # Decoding method
    def decode(self, z):
        return self.decoder(z) # Reconstructs input from latent variable z

    # Forward pass
    def call(self, inputs, training=None):
        mean, logvar = self.encode(inputs) # Encoding
        z = self.reparameterize(mean, logvar) # Reparametrizing
        x_recon = self.decode(z) # Decoding
        return x_recon, mean, logvar, z # Returning reconstructed input, latent distribution parametyers, sampled latent variable
    
    def get_config(self):
        config = super(VAE, self).get_config()  # Get parent config
        config.update({
            'timesteps_per_batch': self.timesteps,
            'n_features': self.n_features,
            'hidden_1': self.hidden_1,
            'hidden_2': self.hidden_2
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
