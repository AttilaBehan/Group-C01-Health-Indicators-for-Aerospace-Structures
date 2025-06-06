import tensorflow as tf

class VAE_Seed():
    vae_seed = 42

# Defines Keras VAE model
class VAE(tf.keras.Model):
    # Contructor method which initializes VAE, hidden_2 = size of latent space, usually smaller than hidden_1
    def __init__(self, timesteps_per_batch, n_features, hidden_1, hidden_2):
        # Calls parent class constructor to initialize model properly
        super(VAE, self).__init__()

        # Storing model parameters
        self.timesteps = timesteps_per_batch
        self.n_features = n_features
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        # Initialization of weights (to improve stability of training, with seed for reproducability)
        initializer = tf.keras.initializers.GlorotUniform(seed=VAE_Seed.vae_seed)

        # Encoder Network 
            # Sequential = linear stack of layers
            # layers: input (with input dim), dense (hidden_1 with signoid activation function), dense (hidden_2 * 2, bc outputs mean and log-variance)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(timesteps_per_batch, n_features)),
            # LSTM processes sequences and returns last output
            tf.keras.layers.LSTM(int(hidden_1), activation='tanh', kernel_initializer=initializer, return_sequences = False),
            # Output mean and log-variance of latent space
            tf.keras.layers.Dense(int(hidden_2) * 2, kernel_initializer=initializer),
        ])


        # Decoder Network
            # Takes latent space (hidden_2) as input, then reconstructs by reversing Encoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(int(hidden_2),)),
            # Expand latent vector to LSTM input size
            tf.keras.layers.Dense(int(hidden_1), activation='relu', kernel_initializer=initializer),
            # Reshape to (batch, timesteps, hidden_1) for LSTM
            tf.keras.layers.RepeatVector(timesteps_per_batch),  # Repeats z `timesteps` times
            # LSTM reconstructs sequences
            tf.keras.layers.LSTM(n_features, activation='tanh', kernel_initializer=initializer, return_sequences=True),
            # No need for Flatten since output is (batch, timesteps, n_features)
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
