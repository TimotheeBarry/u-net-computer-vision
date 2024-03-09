import datetime
import tensorflow as tf

def callback_setup(checkpoint_filepath: str = "", save_logs: bool = False):
    # Fonction qui permet de définir les callbacks pour le modèle
    model_callbacks = []

    '''=== CHECKPOINT DES POIDS ==='''
    if checkpoint_filepath:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='binary_accuracy',
            mode='max',
            save_best_only=True)
        model_callbacks.append(model_checkpoint_callback)

    '''=== LOGS ==='''
    if save_logs:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False)
        model_callbacks.append(tensorboard_callback)

    return model_callbacks