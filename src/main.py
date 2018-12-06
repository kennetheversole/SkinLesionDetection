#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

from skimage.io import imsave

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


#1. Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()





base_skin_dir = os.path.join(os.getcwd(), "data")

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ', 
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

tile_df = pd.read_csv(os.path.join(base_skin_dir, "HAM10000_metadata.csv"), engine='python')


def createGraphs(tile_df):


    # Creating New Columns for better readability


    tile_df.describe(exclude=[np.number])

    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
    fig.savefig('data1.png', dpi=300)

    tile_df = tile_df.drop(tile_df[tile_df.cell_type_idx == 4].iloc[:5000].index)

    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
    tile_df['dx_type'].value_counts().plot(kind='bar')
    fig.savefig('data2.png', dpi=300)


    n_samples = 5
    fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
    for n_axs, (type_name, type_rows) in zip(m_axs, 
                                            tile_df.sort_values(['cell_type']).groupby('cell_type')):
        n_axs[0].set_title(type_name)
        for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
            c_ax.imshow(c_row['image'])
            c_ax.axis('off')
    fig.savefig('category_samples.png', dpi=300)





    rgb_info_df = tile_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v in 
                                    zip(['Red', 'Green', 'Blue'], 
                                        np.mean(x['image'], (0, 1)))}),1)
    gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1)
    for c_col in rgb_info_df.columns:
        rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
    rgb_info_df['Gray_mean'] = gray_col_vec
    rgb_info_df.sample(3)



    for c_col in rgb_info_df.columns:
        tile_df[c_col] = rgb_info_df[c_col].values # we cant afford a copy


    sns.pairplot(tile_df[['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean', 'cell_type']], 
                hue='cell_type', plot_kws = {'alpha': 0.5})



    n_samples = 5
    for sample_col in ['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean']:
        fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
        def take_n_space(in_rows, val_col, n):
            s_rows = in_rows.sort_values([val_col])
            s_idx = np.linspace(0, s_rows.shape[0]-1, n, dtype=int)
            return s_rows.iloc[s_idx]
        for n_axs, (type_name, type_rows) in zip(m_axs, 
                                                tile_df.sort_values(['cell_type']).groupby('cell_type')):

            for c_ax, (_, c_row) in zip(n_axs, 
                                        take_n_space(type_rows, 
                                                    sample_col,
                                                    n_samples).iterrows()):
                c_ax.imshow(c_row['image'])
                c_ax.axis('off')
                c_ax.set_title('{:2.2f}'.format(c_row[sample_col]))
            n_axs[0].set_title(type_name)
        fig.savefig('{}_samples.png'.format(sample_col), dpi=300)



    from skimage.util import montage
    rgb_stack = np.stack(tile_df.\
                        sort_values(['cell_type', 'Red_mean'])['image'].\
                        map(lambda x: x[::5, ::5]).values, 0)
    rgb_montage = np.stack([montage(rgb_stack[:, :, :, i]) for i in range(rgb_stack.shape[3])], -1)
    print(rgb_montage.shape)



    fig, ax1 = plt.subplots(1, 1, figsize = (20, 20), dpi=300)
    ax1.imshow(rgb_montage)
    fig.savefig('nice_montage.png')

    imsave('full_dataset_montage.png', rgb_montage)



def CNN(data_raw):

    class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
    }


    
    features=data_raw.drop(columns=['cell_type_idx'],axis=1)
    target=data_raw['cell_type_idx']
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.15,random_state=1234)

    x_train = np.asarray(x_train_o['image'].tolist())
    x_test = np.asarray(x_test_o['image'].tolist())

    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)

    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)

    x_train = (x_train - x_train_mean)/x_train_std
    x_test = (x_test - x_test_mean)/x_test_std



        

    # Perform one-hot encoding on the labels
    y_train = to_categorical(y_train_o, num_classes = 7)
    y_test = to_categorical(y_test_o, num_classes = 7)

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.15, random_state = 2)


    

    # Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
    x_train = x_train.reshape(x_train.shape[0], *(100,75, 3))
    x_test = x_test.reshape(x_test.shape[0], * (100,75, 3))
    x_validate = x_validate.reshape(x_validate.shape[0], * (100,75, 3))

    input_shape = (100,75, 3)
    num_classes = 7

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
   # model.summary()


    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    epochs = 10 
    batch_size = 10
    history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              class_weight=class_weights,
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
    

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
    print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
    model.save("model.h5")



def main():


    #load and clean the data
    tile_df = pd.read_csv(os.path.join(base_skin_dir, "HAM10000_metadata.csv"), engine='python')


    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes   

    tile_df['sex'].value_counts().plot(kind='bar')

    tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

    tile_df['age'].fillna((tile_df['age'].mean()), inplace=True)

  
    CNN(tile_df)


if __name__ == "__main__":
    main()