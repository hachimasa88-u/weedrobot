import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
#from smallcnn import save_history

# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975


classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy','katabami']

nb_classes = len(classes)
batch_size=32
img_width, img_height = 150, 150
channels=3
train_data_dir = 'train_images'
validation_data_dir = 'test_images'
nb_train_samples = 1260
nb_validation_samples = 180
nb_epoch = 50
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

if __name__ == '__main__':
    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(img_height, img_width, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16_model.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # 学習済みのFC層の重みをロード
    # TODO: ランダムな重みでどうなるか試す
    #top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

    # vgg16_modelはkeras.engine.training.Model
    # top_modelはSequentialとなっている
    # ModelはSequentialでないためadd()がない
    # そのためFunctional APIで二つのモデルを結合する
    # https://github.com/fchollet/keras/issues/4040
    model = Model(vgg16_model.input,top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    # Total params: 16,812,353
    # Trainable params: 16,812,353
    # Non-trainable params: 0
    #model.summary()

    # layerを表示
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # Total params: 16,812,353
    # Trainable params: 9,177,089
    # Non-trainable params: 7,635,264
    #model.summary()

    # TODO: ここでAdamを使うとうまくいかない
    # Fine-tuningのときは学習率を小さくしたSGDの方がよい？
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        color_mode='rgb',
        classes=classes,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        color_mode='rgb',
        classes=classes,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples//batch_size)

    model.save_weights(os.path.join(result_dir, 'finetuning.h5'))
    save_history(history, os.path.join(result_dir, 'history_finetuning.txt'))