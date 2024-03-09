import matplotlib.pyplot as plt


def plot_and_save_history(history, checkpoint_name):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    # summarize history for accuracy

    ax[0].plot(history.history['binary_accuracy'])
    ax[0].plot(history.history['val_binary_accuracy'])
    ax[0].set_title('model accuracy')
    ax[0].legend(['train', 'validation'], loc='upper left')

    # loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].legend(['train', 'validation'], loc='upper left')

    # save fig
    plt.savefig('./backup/' + checkpoint_name + '/train_history.png')