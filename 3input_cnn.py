import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Dropout, Flatten,merge
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate, concatenate
from keras import backend as K
img_rows, img_cols = 75, 75
input_shape = (img_rows, img_cols,1)

input_shape=(512,)
band1_branch_input = Input(shape=input_shape)
band1_branch_shared_conv1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(band1_branch_input)
band1_branch_shared_conv2 = Conv2D(64, (3, 3), activation='relu')(band1_branch_shared_conv1)
band1_branch_shared_pooling = MaxPooling2D(pool_size=(2, 2))(band1_branch_shared_conv2)
band1_branch_shared_dropout = Dropout(0.25)(band1_branch_shared_pooling)
band1_branch_shared_flatten_output = Flatten()(band1_branch_shared_dropout)
band1_branch = Model(band1_branch_input, band1_branch_shared_flatten_output)

input_shape=(512,)
band2_branch_input = Input(shape=input_shape)
band2_branch_shared_conv1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(band2_branch_input)
band2_branch_shared_conv2 = Conv2D(64, (3, 3), activation='relu')(band2_branch_shared_conv1)
band2_branch_shared_pooling = MaxPooling2D(pool_size=(2, 2))(band2_branch_shared_conv2)
band2_branch_shared_dropout = Dropout(0.25)(band2_branch_shared_pooling)
band2_branch_shared_flatten_output = Flatten()(band2_branch_shared_dropout)
band2_branch = Model(band2_branch_input, band2_branch_shared_flatten_output)

merged = concatenate([band1_branch, band2_branch])
merged_classification = Dense(512, activation = 'relu')(merged)
merged_dropout =  Dropout(0.25)(merged_classification)
final_output =  Dense(1, activation = 'sigmoid')(merged_dropout)

model = Model([band1_branch_input, band2_branch_input], final_output)