import numpy as np
import tensorflow as tf
from keras import layers, Model, regularizers
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow import keras
from typing import Tuple
from matplotlib.colors import ListedColormap
from keras import backend as K


"""
The DataPreprocessor class is designed for preprocessing sequence data for FTC-Encoder Model. 

Main Methods:
- collect_data: Collects and processes training data.
- _crop_and_pad_sequences: Crops and pads sequences to a uniform length.
- _crop_sequences: Crops sequences to the nearest multiple of 4.
- _pad_sequences: Pads sequences for the model to ingest.
- create_master_train_data: Creates and prepares the master training dataset.
- _switch_values: Switches values along specified dimensions.
- _concatenate: concatenates training data, appending labels.
- _append_labels: Appends labels to the training data.
- process_test_data: Processes ensuring consistency with training data.

"""
class DataPreprocessor:
    def __init__(self, padding_value=999):
        self.padding_value = padding_value

    def collect_data(self, train, train_summer):
        train_pad, max_length_train = self._crop_and_pad_sequences(train)
        train_summer_pad, _ = self._crop_and_pad_sequences(train_summer, max_length_train)
        
        return train_pad, train_summer_pad

    def _crop_and_pad_sequences(self, data, max_length=None):
        sequence_lengths = self._crop_sequences(data)
        if max_length is None:
            max_length = np.max(sequence_lengths).astype(int)
        padded_data = self._pad_sequences(data, max_length)
        return padded_data, max_length

    def _crop_sequences(self, data):
        lengths = np.empty([data.shape[1]])
        for index in range(data.shape[1]):
            data[0, index] = data[0, index][:, :data[0, index].shape[1] - data[0, index].shape[1] % 4]
            lengths[index] = data[0, index].shape[1]
        return lengths

    def _pad_sequences(self, data, max_length):
        padded_data = np.empty([data.shape[1], max_length, 3])
        for index in range(data.shape[1]):
            padded_data[index, :, :] = np.pad(
                data[0, index], 
                pad_width=((0, 0), (0, max_length - data[0, index].shape[1])), 
                mode='constant', 
                constant_values=self.padding_value
            ).T
        return padded_data

    def create_master_train_data(self, train_pad, train_summer_pad):
        train_pad1 = self._switch_values(train_pad)
        train_summer_pad1 = self._switch_values(train_summer_pad)

        master_train, master_y = self._concatenate(train_pad1, train_summer_pad1)
        xy_train = self._append_labels(master_train, master_y)

        print(xy_train.shape)
        return master_train, xy_train

    def _switch_values(self, data):
        switched_data = data.copy()
        switched_data[:, :, 0], switched_data[:, :, 1] = switched_data[:, :, 1].copy(), switched_data[:, :, 0].copy()
        return switched_data

    def _concatenate(self, train_pad, train_summer_pad):
        master_train = np.concatenate((train_pad, train_summer_pad), axis=0)
        master_y = np.concatenate((np.ones(train_pad.shape[0]), np.zeros(train_summer_pad.shape[0])), axis=0)

        non_zero_mask = (master_train != self.padding_value)
        non_zero_mean = np.mean(master_train[non_zero_mask])
        non_zero_std = np.std(master_train[non_zero_mask])

        master_train = master_train
        master_train[~non_zero_mask] = self.padding_value

        return master_train, master_y

    def _append_labels(self, master_train, master_y):
        return np.concatenate(
            (master_train, np.broadcast_to(master_y[:, None, None], master_train.shape[:-1] + (1,))),
            axis=-1
        )
    
    def process_test_data(self, data_smap_un):
        # Crop sequences
        for index in range(data_smap_un.shape[0]):
            data_smap_un[index, 0] = data_smap_un[index, 0][:, :data_smap_un[index, 0].shape[1] - data_smap_un[index, 0].shape[1] % 4]
        
        # Calculate max length
        x = np.empty([data_smap_un.shape[0]])
        for index in range(data_smap_un.shape[0]):
            x[index] = np.array(data_smap_un[index, 0].shape[1])
        max_length_test = np.max(x).astype(int)
        
        # Pad sequences
        test_ismn = np.empty([data_smap_un.shape[0], max_length_test, 3])
        for index in range(data_smap_un.shape[0]):
            test_ismn[index, :, :] = np.pad(
                data_smap_un[index, 0], 
                pad_width=((0, 0), (0, max_length_test - data_smap_un[index, 0].shape[1])), 
                mode='constant', 
                constant_values=self.padding_value
            ).T
        
        # Masking
        non_zero_mask = (test_ismn != self.padding_value)
        
        test_data = test_ismn
        test_data[~non_zero_mask] = self.padding_value
        
        # Switching values along the third dimension
        switched_data = test_data.copy()  # Make a copy to avoid modifying the original array
        switched_data[:, :, 0], switched_data[:, :, 1] = switched_data[:, :, 1].copy(), switched_data[:, :, 0].copy()
        
        return switched_data


"""
The AutoencoderModel class constructs FTC-Encoder Model for sequence data. 
This class includes methods to build, retrieve, and summarize the model, designed 
to encode input sequences into a lower-dimensional representation and then decode 
them back to the original format.

Main Methods:
- _build_model: Constructs the autoencoder model by defining the encoder and decoder networks.
- get_model: Returns the constructed autoencoder model.
- summary: Prints the summary of the autoencoder model architecture.
"""

class AutoencoderModel:
    def __init__(self, dropout_rate, filter_size, kernel_size, l2_reg_rate):
        self.dropout_rate = dropout_rate
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.l2_reg_rate = l2_reg_rate
        self.model = self._build_model()
        
    def _build_model(self):
        encoder_input = layers.Input(shape=(None, 3))

        # Encoder
        encoder_masking = layers.Masking(mask_value=999)(encoder_input)
        encoder_conv1d_1 = layers.Conv1D(
            filters=self.filter_size * 2,
            kernel_size=self.kernel_size,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg_rate),
        )(encoder_masking)
        encoder_batchnorm_1 = BatchNormalization()(encoder_conv1d_1)
        encoder_dropout_1 = layers.Dropout(rate=self.dropout_rate)(encoder_batchnorm_1)
        
        encoder_conv1d_2 = layers.Conv1D(
            filters=self.filter_size,
            kernel_size=self.kernel_size,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg_rate),
        )(encoder_dropout_1)
        encoder_batchnorm_2 = BatchNormalization()(encoder_conv1d_2)

        # Decoder
        decoder_conv1d_transpose_1 = layers.Conv1DTranspose(
            filters=self.filter_size,
            kernel_size=self.kernel_size,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg_rate),
        )(encoder_batchnorm_2)
        decoder_batchnorm_1 = BatchNormalization()(decoder_conv1d_transpose_1)
        decoder_dropout_1 = layers.Dropout(rate=self.dropout_rate)(decoder_batchnorm_1)

        decoder_conv1d_transpose_2 = layers.Conv1DTranspose(
            filters=self.filter_size * 2,
            kernel_size=self.kernel_size,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg_rate),
        )(decoder_dropout_1)
        decoder_batchnorm_2 = BatchNormalization()(decoder_conv1d_transpose_2)

        decoder_output = layers.Conv1DTranspose(
            filters=3,
            kernel_size=self.kernel_size,
            padding="same",
        )(decoder_batchnorm_2)

        # Define the autoencoder model (encoder + decoder)
        autoencoder_model = Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
        return autoencoder_model

    def get_model(self):
        return self.model

    def summary(self):
        self.model.summary()


def contrastive_loss_function(xy_true, x_pred, mask_value=999, epsilon=1e-7):
    """
    Custom contrastive loss function with masking.
    
    Args:
        xy_true (tf.Tensor): True values tensor, including labels.
        x_pred (tf.Tensor): Predicted values tensor.
        mask_value (int, optional): Value to mask in the true values tensor. Defaults to 999.
        epsilon (float, optional): Small value to avoid division by zero and log of zero. Defaults to 1e-7.
    
    Returns:
        tf.Tensor: Computed loss value.
    """
    
    # Extract the true values and labels from the input tensor
    x_true = xy_true[:, :, :3]
    t = xy_true[:, 1, 3]
    
    # Ensure x_true and x_pred have the same data type
    x_true = K.cast(x_true, dtype=K.floatx())
    x_pred = K.cast(x_pred, dtype=K.floatx())
    
    # Create a mask to identify padded values
    mask = K.not_equal(x_true, mask_value)
    
    # Compute the mean squared error with masking
    mse = K.mean(K.square((x_true - x_pred) * K.cast(mask, dtype=K.floatx())), axis=-1)
    
    # Clip mse to avoid numerical instabilities
    mse = K.clip(mse, epsilon, 1e10)
    
    # Compute the error term
    err = t[:, np.newaxis] * mse - (1 - t[:, np.newaxis]) * tf.math.log1p(-K.exp(-mse))
    
    # Return the sum of errors
    return K.sum(err, axis=-1)


def Accuracy(xy_true, x_pred, mask_value=999, epsilon=1e-7):
    """
    Custom contrastive loss function with masking.
    
    Args:
        xy_true (tf.Tensor): True values tensor, including labels.
        x_pred (tf.Tensor): Predicted values tensor.
        mask_value (int, optional): Value to mask in the true values tensor. Defaults to 999.
        epsilon (float, optional): Small value to avoid division by zero and log of zero. Defaults to 1e-7.
    
    Returns:
        tf.Tensor: Computed loss value.
    """
    
    # Extract the true values and labels from the input tensor
    x_true = xy_true[:, :, :3]
    t = xy_true[:, 1, 3]
    
    # Ensure x_true and x_pred have the same data type
    x_true = K.cast(x_true, dtype=K.floatx())
    x_pred = K.cast(x_pred, dtype=K.floatx())
    
    # Create a mask to identify padded values
    mask = K.not_equal(x_true, mask_value)
    
    # Compute the mean squared error with masking
    mse = K.mean(K.square((x_true - x_pred) * K.cast(mask, dtype=K.floatx())), axis=-1)
    mse = K.clip(mse, epsilon, 1e10)

    prob = K.exp(-mse)
    # Clip mse to avoid numerical issues
    
    # Compute the error term
    pred_label = prob > 0.5
    pred_label = K.cast(pred_label, dtype=tf.int32)
    

    # Replicate true_label to match pred_label's shape
    true_label_replicated = tf.tile(t[:, tf.newaxis], [1, prob.shape[1]])
    true_label_replicated = K.cast(true_label_replicated, dtype=tf.int32)
   
    # Create a mask for the agreement calculation
    cts = tf.not_equal(x_true[:, :, 1], mask_value)
    cts = K.cast(cts, dtype = tf.int32)
    # Calculate agreement percentage
    #correct_predictions = tf.reduce_sum(tf.cast(pred_label & cts, dtype= tf.int32) == tf.cast(true_label_replicated & cts, dtype=tf.int32), axis=1)
    correct_predictions = tf.reduce_sum(tf.cast((pred_label) == (true_label_replicated), dtype=tf.float32) * K.cast(cts, dtype=K.floatx()), axis=1)

    total_predictions = tf.reduce_sum(cts, axis=1)
    agreement_percentage = tf.cast(correct_predictions, dtype = K.floatx()) / tf.cast(total_predictions, dtype = K.floatx()) * 100
    return tf.reduce_mean(agreement_percentage)

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 30
) -> tf.keras.callbacks.History:
    """
    Train the given model with the specified parameters and return the training history.

    Args:
        model (tf.keras.Model): The model to be trained.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        y_test (np.ndarray): Testing labels.
        epochs (int, optional): Number of epochs to train. Defaults to 200.
        batch_size (int, optional): Size of the training batches. Defaults to 64.
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 30.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1  # 0: Suppresses output for each epoch

    )
    return history

def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training and validation loss from the training history.

    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


def compute_accuracy(autoencoder_model, X_data, y_data):
    """
    Compute accuracy metrics for the given model and data.

    Args:
        autoencoder_model (tf.keras.Model): Trained autoencoder model.
        X_data (np.ndarray): Input data.
        y_data (np.ndarray): True labels.

    Returns:
        float: Agreement percentage for the data.
    """
    reconst = autoencoder_model.predict(X_data)
    mask = K.not_equal(X_data, 999)
    mse = keras.backend.mean(keras.backend.square((X_data - reconst) * K.cast(mask, dtype=K.floatx())), axis=-1)
    prob = np.exp(-mse)
    original_array = y_data[:, 1, 3]
    true_label_array = np.tile(original_array[:, None], X_data.shape[1])

    agreement_percentage = np.empty(X_data.shape[0])

    for n in range(X_data.shape[0]):
        window_size = 1
        smoothed_prob = np.convolve(prob[n, :], np.ones(window_size) / window_size, mode='same')
        pred_label = smoothed_prob > 0.5
        true_label = true_label_array[n, :]

        cts = ~(X_data[n, :, 1] == 999)
        agreement_percentage[n] = np.sum(pred_label[cts] == true_label[cts]) / np.sum(cts) * 100

    return np.mean(agreement_percentage)

def compute_train_val_accuracy(autoencoder_model, X_train, y_train, X_test, y_test):
    """
    Compute training and validation accuracy for the autoencoder model.

    Args:
        autoencoder_model (tf.keras.Model): Trained autoencoder model.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        y_test (np.ndarray): Testing labels.

    Returns:
        tuple: Training and testing accuracy.
    """
    training_accuracy = compute_accuracy(autoencoder_model, X_train, y_train)
    print(f'Training Accuracy: {training_accuracy:.2f}%')

    testing_accuracy = compute_accuracy(autoencoder_model, X_test, y_test)
    print(f'Testing Accuracy: {testing_accuracy:.2f}%')

    return training_accuracy, testing_accuracy




def visualize_probability_time_series(n, original_data, prob, data_var, fs=20, start_timeid=0, end_timeid=365):

    """
    Visualize the probability of frozen events and corresponding time series data.

    Args:
        n (int): The index of the data sample to be visualized.
        original_data (np.ndarray): A numpy array containing the original time series data.
        prob (np.ndarray): A numpy array containing the probability of frozen events.
        data_var (np.ndarray): A numpy array containing additional data variables for comparison.
        fs (int, optional): Font size for the plot labels and titles. Defaults to 20.
        start_timeid (int, optional): The starting index of the time series to be visualized. Defaults to 0.
        end_timeid (int, optional): The ending index of the time series to be visualized. Defaults to 365.

    Returns:
        Accuracy, Plot
    """
    cmap_red_blue = ListedColormap(['#d73027', '#4575b4'])  # Red and Blue shades
    cmap_brown_green = ListedColormap(['#8c510a', '#01665e'])  # Brown and Green shades

    # Create a single figure with two subplots sharing the x-axis
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # First subplot
    ax1.plot(original_data[n, 0][0, start_timeid:end_timeid], color=cmap_red_blue(1), label='V-pol', linewidth=2)
    ax1.plot(original_data[n, 0][1, start_timeid:end_timeid], color=cmap_red_blue(0), label='H-pol', linewidth=2)
    ax1.set_xlim(0, len(original_data[n, 0][1, start_timeid:end_timeid]) - 1)
    ax1.set_ylabel(r'$T_{\rm B}^{p}$', fontsize=fs)
    ax1.set_title('FTC-Encoder', fontsize=fs, fontweight="bold")
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(1-prob[n, start_timeid:end_timeid], color='black', label=r'$P(F)$', linewidth=2)
    ax2.set_ylabel(r'$P(F)$', fontsize=fs)
    ax2.bar(range(len(prob[n, start_timeid:end_timeid])), prob[n, start_timeid:end_timeid] < 0.5, color='lightblue', alpha=0.2, label='Frozen')
    pred_label = prob[n, start_timeid:end_timeid] < 0.5
    ax2.set_ylim(0,1)
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.grid(False)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1.01, 1.2), fontsize=fs-5, ncol=4,  handlelength=1.5, handletextpad=0.4, labelspacing=0.2, borderpad=0.3)

    # Second subplot
    ax3.plot(data_var[n, 0][1, start_timeid:end_timeid], color=cmap_brown_green(1), label=r'$T_{\mathrm{air}}$', linewidth=2)
    ax3.plot(data_var[n, 0][0, start_timeid:end_timeid], color=cmap_brown_green(0), label=r'$T_\mathrm{g}$', linewidth=2)
    ax3.set_xlabel('Days', fontsize=fs)
    ax3.set_ylabel('Temperature', fontsize=fs)
    ax3.set_ylim([230, 300])
    ax3.set_xlim(0, len(original_data[n, 0][1, start_timeid:end_timeid]) - 1)
    ax3.set_title('ERA5 Temperature Data', fontsize=fs,fontweight="bold")
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.grid(True)

    ax4 = ax3.twinx()
    ax4.bar(range(len(data_var[n, 0][0, start_timeid:end_timeid])), (data_var[n, 0][1, start_timeid:end_timeid] < 273.1), color='skyblue', alpha=0.2, label='Frozen')
    true_label = data_var[n, 0][0, start_timeid:end_timeid] < 273.1
    ax4.set_ylim(0,1)
    ax4.set_ylabel(r'ERA5 FT state', fontsize=fs)
    ax4.tick_params(axis='both', labelsize=fs)
    ax4.grid(False)

    # Combine legends
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right', bbox_to_anchor=(1.01, 1.2), fontsize=fs-5, ncol=3,  labelspacing=0.05)
    ax3.axhline(y=273.15, color='black', linestyle='--', linewidth=2, label='273.15 K')

    # Calculate Accuracy
    cts = ~np.isnan(prob[n, start_timeid:end_timeid]) & ~np.isnan(data_var[n, 0][0, start_timeid:end_timeid])
    agreement_percentage = np.sum(pred_label[cts] == true_label[cts]) / np.sum(cts) * 100
    print(f'Accuracy: {agreement_percentage:.2f}%')

    plt.tight_layout()
    plt.show()

