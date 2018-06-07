import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from keras.layers import concatenate
import cv2
import time
import multiprocessing as mp
import psutil

#=======================================
#Load the data Input.
#Split into X and Y.
#30% of the data goes into testing.
#=======================================

data = pd.read_csv('coinData.csv', sep=',', engine='c', header=None, na_filter=False, low_memory=True)

X_total = data.iloc[:,1:]
y_total = data.iloc[:,:1]
x_train, x_test, y_train, y_test = train_test_split(X_total, y_total, test_size = 0.3)

#=======================================
#Each image is 80 x 80 = 6400 grayscale 
#pixels ranging in value (0,255).
#Reshape data to fit images.
#Split into seperate groups, we have 
#50 groups with 6453 images.
#=======================================

x_train = x_train.values.reshape(-1, 80, 80, 1).astype('float32') / 255.
x_test = x_test.values.reshape(-1, 80, 80, 1).astype('float32') / 255.

y_train = y_train.values.astype('int')
y_test = y_test.values.astype('int')

print('Training', x_train.shape, x_train.max())
print('Testing', x_test.shape, x_test.max())

train_groups = [x_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_train)]

sumTrain = [x.shape[0] for x in train_groups]
sumTrain = sum(sumTrain)
sumTest = [x.shape[0] for x in test_groups]
sumTest = sum(sumTest)

print('train groups:', [x.shape[0] for x in train_groups], "Sum:",sumTrain)
print('test groups:', [x.shape[0] for x in test_groups], "Sum:",sumTest)

#=======================================
#For the siamese network we need to split input
#into two groups X & Y and then the score between them
#if images are in the same group similarity is assumed
#as 1 else 0
#=======================================

def random_batch(in_groups, batch_halfsize = 8):
    out_img_x, out_img_y, out_score = [], [], []
    all_groups = list(range(len(in_groups)))

    for match_group in [True, False]:
        group_idx = np.random.choice(all_groups, size = batch_halfsize)
        out_img_x += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
        
        if match_group:
            b_group_idx = group_idx
            out_score += [1]*batch_halfsize
        else:
            # anything but the same group
            non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 
            b_group_idx = non_group_idx
            out_score += [0]*batch_halfsize
            
        out_img_y += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]
            
    return np.stack(out_img_x,0), np.stack(out_img_y,0), np.stack(out_score,0)

#=======================================
#Prepare the images for the nn.
#=======================================
print("Shape:",x_train.shape[1:])
img_in = Input(shape = x_train.shape[1:], name = 'FeatureNet_ImageInput')
n_layer = img_in


#=======================================
#Parse the images through a network to generate
#feature data, starts of randoml initialised.
#20 layers including the input.
#=======================================

for i in range(1):
    n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = MaxPool2D((2,2))(n_layer)

n_layer = Flatten()(n_layer)
n_layer = Dense(32, activation = 'linear')(n_layer)
n_layer = Dropout(0.5)(n_layer)
n_layer = BatchNormalization()(n_layer)
n_layer = Activation('relu')(n_layer)

feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')
feature_model.summary()

#=======================================
#Parse images x and y into the model for features
#=======================================

img_x_in = Input(shape = x_train.shape[1:], name = 'ImageX_Input')
img_y_in = Input(shape = x_train.shape[1:], name = 'ImageY_Input')
img_x_feat = feature_model(img_x_in)
img_y_feat = feature_model(img_y_in)

#=======================================
#We've applied the feature model to both
#images, now we merge them together as part
#of a simple Siamese model, including the merge
# we have 8 layers.
#=======================================

combined_features = concatenate([img_x_feat, img_y_feat], name = 'merge_features')
combined_features = Dense(4, activation = 'linear')(combined_features)
combined_features = BatchNormalization()(combined_features)
combined_features = Activation('relu')(combined_features)
combined_features = Dense(1, activation = 'sigmoid')(combined_features)

similarity_model = Model(inputs = [img_x_in, img_y_in], outputs = [combined_features], name = 'Similarity_Model')
similarity_model.summary()

#=======================================
#Configure the learning process
#=======================================

similarity_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['mae'])

#=======================================
#Take a random sample and test accuracy
#to let us know how it's going
#=======================================

def model_test(nb_examples = 3):
    img_x, img_y, img_sim = random_batch(test_groups, nb_examples)
    pred_sim = similarity_model.predict([img_x, img_y])
    fig, m_axs = plt.subplots(2, img_x.shape[0], figsize = (12, 6))
    
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(img_x, img_y, img_sim, pred_sim, m_axs.T):
        ax1.imshow(c_a[:,:,0])
        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*c_d))
        ax1.axis('off')
        ax2.imshow(c_b[:,:,0])
        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p_d))
        ax2.axis('off')

    return fig

#=======================================
#Function to generate the siamese data
#=======================================

def siam_gen(in_groups, batch_size = 32):
    while True:
        img_x, img_y, pv_sim = random_batch(train_groups, batch_size//2)
        yield [img_x, img_y], pv_sim

#=======================================
#Constant validation group to give us as 
#a frame of reference
#=======================================

valid_x, valid_y, valid_sim = random_batch(test_groups, 1024)
loss_history = similarity_model.fit_generator(siam_gen(train_groups), steps_per_epoch = 500, validation_data=([valid_x, valid_y], valid_sim), epochs = 1, verbose = True)



#=======================================
#Give an example of the output after training
#=======================================

def run_predict():
    fig = model_test()
    fig.show()
    plt.show()


def monitor(target):
    if __name__ == '__main__':
        worker_process = mp.Process(target=target)
        worker_process.start()
        p = psutil.Process(worker_process.pid)

        # log cpu usage of `worker_process` every 10 ms
        cpu_percents = []
        while worker_process.is_alive():
            cpu_percents.append(p.cpu_percent())
            time.sleep(0.01)

        worker_process.join()
        return cpu_percents

cpu_percents = monitor(target=run_predict)

print("CPU:",cpu_percents)
#=======================================
#Save the models
#=======================================

#feature_model.save('feature.h5')
#similarity_model.save('similarity.h5')