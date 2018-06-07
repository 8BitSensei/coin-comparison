import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.collections as clc
from matplotlib.path import Path
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from keras.layers import concatenate
from keras.models import load_model
import cv2
from sklearn.manifold import TSNE
import math
from keras.utils import plot_model

#=======================================
#Load data in
#=======================================
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
data = pd.read_csv('coinData.csv', sep=',', engine='c', header=None, na_filter=False, low_memory=True)

X_total = data.iloc[:,1:]
y_total = data.iloc[:,:1]
x_train, x_test, y_train, y_test = train_test_split(X_total, y_total, test_size = 0.3)


x_train = x_train.values.reshape(-1, 80, 80, 1).astype('float32') / 255.
x_test = x_test.values.reshape(-1, 80, 80, 1).astype('float32') / 255.

y_train = y_train.values.astype('int')
y_test = y_test.values.astype('int')

print('Training', x_train.shape, x_train.max())
print('Testing', x_test.shape, x_test.max())

train_groups = [x_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_train)]

#=======================================
#Load in the feature models 
#=======================================

feature_model = load_model('feature.h5')
similarity_model = load_model('similarity.h5')

plot_model(feature_model, to_file='feature.png')
plot_model(similarity_model, to_file='similarity.png')

obj_categories = ['Anonymous',
'Anonymous 0', 
'Anonymous 1', 
'Anonymous 2', 
'Anonymous 3', 
'Anonymous 4', 
'Anonymous 5', 
'Anonymous 6', 
'Anonymous 7', 
'Anonymous 8', 
'Anonymous 9', 
'Anonymous 10', 
'Anonymous 11', 
'Anonymous 12', 
'Anonymous 13', 
'Anonymous 14', 
'Tiberius - 14',
'Caratacus - 50',
'Vespasian - 69',
'Hadrian - 117',
'Marcus Aurelieus - 161',
'Septimus Severus - 193',
'Elagabalus - 218',
'Severus Alexander - 222',
'Gordian III - 242',
'Philip I - 244',
'Philip II - 247',
'Decius - 249',
'Trebonianus Gallus - 251',
'Volusian - 251',
'Diocletian - 284',
'Maximian - 285',
'Carausius - 286',
'Galerius - 293',
'Maximinus II - 305',
'Constantine the Great - 306',
'Constantinus I - 306',
'Licinius I - 308',
'Constantine II - 317',
'Crispus - 320',
'Constantinus II - 324',
'Julian the Apostate - 355',
'Valens - 364',
'Gratian - 367',
'Valentinian II - 375',
'Theodosius I - 379',
'Magnus Maximus - 385']

names = np.array(obj_categories)

obj_cultures = ['Native','Native','Native','Roman','Native','Native','Native','Native','Native','Native','Native','Native','Native','Native','Native','Native','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman','Roman']

x_test_features = feature_model.predict(x_test, verbose = True, batch_size=128)

tsne_obj = TSNE(n_components=2,init='pca',random_state=101,method='barnes_hut',n_iter=500,verbose=2)

tsne_features = tsne_obj.fit_transform(x_test_features)

c_colors = plt.cm.rainbow(np.linspace(0, 1, 34))
n_colours = plt.cm.rainbow(np.linspace(0, 1, 16))
plt.figure(figsize=(4, 4))

#sc = []
#c = np.random.randint(1,5,size=15)

#norm = plt.Normalize(1,4)
#cmap = plt.cm.RdYlGn

#fig,ax = plt.subplots()
cds = []

for c_group, (c_color, c_label, c_culture) in enumerate(zip(c_colors, obj_categories, obj_cultures)):
	
	x = tsne_features[np.where(y_test == c_group), 0]
	y = tsne_features[np.where(y_test == c_group), 1]

	if c_culture == 'Roman':
		plt.scatter(tsne_features[np.where(y_test == c_group), 0],
                tsne_features[np.where(y_test == c_group), 1],
                marker='o',
                color=c_color,
                linewidth='0.5',
                alpha=0.4,
                label=c_label)
	else:
		plt.scatter(tsne_features[np.where(y_test == c_group), 0],
                tsne_features[np.where(y_test == c_group), 1],
                marker='x',
                color=c_color,
                linewidth='1',
                alpha=1,
                label=c_label)
	
	cds.append([c_label,x,y])
	
	

#cds[0] - Group
#cds[0][0] - Group Label
#cds[0][1] - Co-ordinates
#cds[0][1][0] - Co-ordinates one
#cds[0][1][0][0] - Co-ordinates X
#cds[0][1][0][1] - Co-ordinates Y

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE on Testing Samples')
plt.legend(loc='best', prop={'size': 6})
#plt.annotate(str(cds[0][0]),(cds[0][1],cds[0][2]))

plt.show()