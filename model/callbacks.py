from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
import neptune
import config
class NeptuneMonitor(Callback):
    def __init__(self, n_batch):
        super().__init__()
        self.n = n_batch
        self.current_epoch = 0

    # def on_batch_end(self, batch, logs={}):
    #     x = (self.current_epoch * self.n) + batch
    #     neptune.send_metric(channel_name='batch end accuracy', x=x, y=logs['acc'])
    #     neptune.send_metric(channel_name='batch end loss', x=x, y=logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        val_acc = logs['val_acc']
        val_loss = logs['val_loss']
        neptune.send_metric('validation acc', epoch, val_acc)
        neptune.send_metric('validation loss', epoch, val_loss)
        self.current_epoch += 1

def get_callbacks(model_path):
    # reduces learning rate on plateau
    lr_reducer = ReduceLROnPlateau(factor=0.7,
                                   cooldown=10,
                                   patience=10, verbose=1,
                                   min_lr=0.1e-5)
    # model checkpoint callback
    mode_autosave = ModelCheckpoint(model_path, monitor='val_acc',
                                    mode='max', save_best_only=True, verbose=1, period=1)

    # stop learining as metric on validaton stop increasing
    early_stopping = EarlyStopping(monitor='val_acc', patience=int(10), verbose=1, mode='auto')

    # initialize NeptuneMonitor
    neptune_monitor = NeptuneMonitor(config.BATCH_SIZE)
    callbacks = [mode_autosave, lr_reducer, neptune_monitor, early_stopping]

    return callbacks