import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, Flatten
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

N = 506
nrecs = 3005

# Create Brownian Motion
x0 = np.random.randn(nrecs,N)*0.02
x1 = np.random.randn(nrecs,N)*0.01
x2 = np.random.randn(nrecs,N)*0.005
v0 = np.cumsum(x0,axis=1)
v1 = np.cumsum(x1,axis=1)
v2 = np.cumsum(x2,axis=1)

# Create Drift Components
b0 = 0 #np.arange(N)*0.005
b1 = 0 #b0 + 0.001*b0*b0
b2 = 0 #b0 + 0.002*b0*b0

p0 = v0+b0
p1 = v1+b1
p2 = v2+b2

print(p0.shape)

for k in range(nrecs):
    nzlen0 = np.random.randint(10,40)
    p0[k,-nzlen0:] = 0
    nzlen1 = np.random.randint(10,40)
    p1[k,-nzlen1:] = 0
    nzlen2 = np.random.randint(10,40)
    p2[k,-nzlen2:] = 0
    
plt.figure()
plt.subplot(231), plt.plot(b0)
plt.title('class 1')
plt.subplot(234), plt.plot(p0.T)
plt.subplot(232), plt.plot(b1)
plt.title('class 2')
plt.subplot(235), plt.plot(p1.T)
plt.subplot(233), plt.plot(b2)
plt.title('class 3')
plt.subplot(236), plt.plot(p2.T)
plt.show()

plt.figure()
plt.plot(p0.T,'b')
plt.plot(p1.T,'r')
plt.plot(p2.T,'g')
plt.show()

#one
plt.subplot(311)
plt.hist(np.diff(p0[0,:-100]),bins=20)
plt.subplot(312)
plt.hist(np.diff(p1[0,:-100]),bins=20)
plt.subplot(313)
plt.hist(np.diff(p2[0,:-100]),bins=20)

print("maximum value of data = ", np.max([p0,p1,p2]))
print("minimum value of data = ", np.min([p0,p1,p2]))

tX = np.concatenate((p0,p1),axis=0)
X = np.concatenate((tX,p2),axis=0)
print("X shape = ", X.shape)
print("minimum value of data = ",np.min(X))
print("maximum value of data = ",np.max(X))
 
print("mean value of data = ",np.mean(X))
print("Scaling data by min and max")

rsX = 2*(X-np.min(X))/(np.max(X)-np.min(X)) - 1
print("scaled minimum of data = ", np.min(rsX))
print("scaled maximum of data = ", np.max(rsX))

ncategories = 3
y0 = np.zeros((nrecs,N))
y1 = np.ones((nrecs,N))
y2 = np.ones((nrecs,N))*2

tY = np.concatenate((y0,y1),axis=0)
Y = np.concatenate((tY,y2),axis=0)

# Create a set of training sequences
x_train, x_test, y_train, y_test = train_test_split(rsX,Y,test_size=0.2,random_state=42)

maxseqlen = x_train.shape[1]
nrecords_train = x_train.shape[0]
nrecords_test = x_test.shape[0]

###############################
# Configure the training data #
###############################

flatTrain = x_train.reshape(nrecords_train*maxseqlen,1)
print("flatTrain shape = ",flatTrain.shape)

#minmaxscaler = MinMaxScaler()

#trajTrainNormPre = minmaxscaler.fit_transform(flatTrain)

#trajTrainNorm = trajTrainNormPre.reshape(nrecords_train,maxseqlen,1)
trajTrainNorm = flatTrain.reshape(nrecords_train,maxseqlen,1)
print("trajTrainNorm shape = ",trajTrainNorm.shape)

#################################
# Configure the training labels #
#################################
ltr3 = y_train.reshape(nrecords_train*maxseqlen)

one_hot_encoded = to_categorical(ltr3)
print("one_hot_encoded shape = ",one_hot_encoded.shape)

labelTrain = one_hot_encoded.reshape(nrecords_train,maxseqlen,ncategories)
print("labelTrain shape = ", labelTrain.shape)

###########################
# Configure the Test Data #
###########################
flatTest = x_test.reshape(nrecords_test*maxseqlen,1)
#trajTestNormPre = minmaxscaler.fit_transform(flatTest)
#trajTestNorm = trajTestNormPre.reshape(nrecords_test,maxseqlen,1)
trajTestNorm = flatTest.reshape(nrecords_test,maxseqlen,1)

#############################
# Configure the Test Labels #
#############################
lts3 = y_test.reshape(nrecords_test*maxseqlen)
one_hot_encoded2 = to_categorical(lts3)
labelTest = one_hot_encoded2.reshape(nrecords_test,maxseqlen,ncategories)

print(trajTrainNorm.shape)
# Plot out distributions of scaled data
plt.plot(np.squeeze(trajTrainNorm.T))
plt.show()

wsize = 40
print('trajTrainNorm.shape = ',trajTrainNorm.shape)
print('labelTrain.shape = ',labelTrain.shape)

bpad = int(np.round(wsize*(np.ceil(trajTrainNorm.shape[1]/wsize)))) - trajTrainNorm.shape[1]
bpad2 = int(np.round(wsize*(np.ceil(labelTrain.shape[1]/wsize)))) - labelTrain.shape[1]
print('bpad = ',bpad)
print('bpad2 = ',bpad2)

bufferarray = np.zeros((trajTrainNorm.shape[0],bpad,trajTrainNorm.shape[2]))
bufferarray2 = np.zeros((labelTrain.shape[0],bpad2,labelTrain.shape[2]))
print('bufferarray.shape = ',bufferarray.shape)
print('bufferarray2.shape = ',bufferarray2.shape)

ntrajTrainNorm = np.concatenate((trajTrainNorm,bufferarray),axis=1)
nlabelTrain = np.concatenate((labelTrain,bufferarray2),axis=1)
print('ntrajTrainNorm.shape = ', ntrajTrainNorm.shape)
print('nlabelTrain.shape = ', nlabelTrain.shape)

s0 = ntrajTrainNorm.shape[0]*wsize
s1 = int(ntrajTrainNorm.shape[1]/wsize)
s2 = ntrajTrainNorm.shape[2]
nbatch_train = np.reshape(ntrajTrainNorm,(s0,s1,s2))
print('nbatch_train.shape = ',nbatch_train.shape)

u0 = nlabelTrain.shape[0]*wsize
u1 = int(nlabelTrain.shape[1]/wsize)
u2 = nlabelTrain.shape[2]
nlabel_train = np.reshape(nlabelTrain, (u0,u1,u2))
print('nlabel_train.shape = ',nlabel_train.shape)

# Remove segments that are equal across the window 
# stemming from zero padding since the original sequences
# were not all equal and should remain through
# this filtering step

# Data
dsbatch_train = nbatch_train[0,:,0]
dsbatch_train = np.expand_dims(dsbatch_train,axis=0)

# labels
dslabel_train = nlabel_train[0,:,:]
dslabel_train = np.expand_dims(dslabel_train,axis=0)

for k in range(nrecs):
    if not(np.all((nbatch_train[k,:,0] == nbatch_train[k,0,0]))):
        
        tmpbatch = np.expand_dims(nbatch_train[k,:,0],axis=0)
        tmplabel = np.expand_dims(nlabel_train[k,:,:],axis=0)

        dsbatch_train = np.concatenate((dsbatch_train,tmpbatch),axis=0)
        dslabel_train = np.concatenate((dslabel_train,tmplabel),axis=0)

dsbatch_train = np.expand_dims(dsbatch_train,axis=2)
print('dsbatch_train.shape = ',dsbatch_train.shape)
print('dslabel_train.shape = ',dslabel_train.shape)

############################
# Specify the Model Layers #
############################
#def ResNet50(input_shape = (13,1),classes=3):
#    """ 
#    Implementation of the popular ResNet50 architecture:
#    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER 
    
#    Arguments:
#    input_shape -- shape of the images of the dataset
#    classes -- integer, number of classes
    
#    Returns:
#    model -- a Model() instance in Keras
#    """
    
#    # Define the input as a tensor with shape input_shape
#    X_input = Input(input_shape)
    
    # Zero-Padding
#    X = ZeroPadding2D((3,1))(X_input)
    
    # Stage 1
#    X = Conv1D()
#inputs = tf.keras.Input(name='inputs',shape=(None,1))
layer = tf.keras.layers.Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu',input_shape=(None,1))(inputs)
layer = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(layer)
layer = tf.keras.layers.Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(layer)
layer = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(layer)
layer = tf.keras.layers.Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(layer)
layer = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(layer)
layer = tf.keras.layers.Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(layer)
layer = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(layer)
layer = Dense(ncategories,activation='softmax')(layer)

model = tf.keras.Model(inputs=inputs,outputs=layer)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])   

###################
# Train the Model #
###################
print('Training the model')                     
history = model.fit(dsbatch_train, dslabel_train, epochs=200, validation_split = 0.4)
#, validation_data=(trajTestNorm, labelTest))
dsbatch_train.shape

plt.subplot(211)
plt.plot(history.history['loss'],'b',label='training loss')
plt.plot(history.history['val_loss'],'g',label='validation_loss')
plt.legend()
plt.subplot(212)
plt.plot(history.history['accuracy'],'b',label='training_accuracy')
plt.plot(history.history['val_accuracy'],'g',label='validation_accuracy')
plt.legend()
history.history.keys()

model.save('sequential_learning_3classes_LSTMonly_model.h5')
#score, acc = model.evaluate(trajTestNorm,labelTest)

#print('Test score: ', score)
#print('Test accuracy: ', acc)

def window_batch(trajdata,win):
    
    #print(trajTestNorm[300].T.shape)
    #tmp = np.expand_dims(np.squeeze(trajTestNorm[300,:,:]),axis=0)

    print(range((trajdata.size)))
    for k in range((trajdata.size)-win+1):
        if k==0:
            v = trajdata[:,k:k+win]
            print(v.shape)
        else:
            #print(tmp[:,k:k+win].shape)
            #print(tmp[:,k:k+win])
            v = np.concatenate((v,trajdata[:,k:k+win]),axis=0)

    ve = np.expand_dims(v,axis=2)
    
    return ve
    
idx = 300
rolling_windows = 10
print('trajTestNorm[idx].T.shape = ',trajTestNorm[idx].T.shape)
tmp = np.expand_dims(np.squeeze(trajTestNorm[idx,:,:]),axis=0)
print('tmp.shape = ',tmp.shape)
ve = window_batch(tmp,rolling_windows)
print("tmp.shape = ",tmp.shape)
print("ve.shape = ",ve.shape)

y_pred = model.predict(ve)
print("y_pred.shape = ",y_pred.shape)

nypred = np.reshape(y_pred,(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2]))
print("nypred.shape = ",nypred.shape)

plt.subplot(321)
plt.plot(x_test[idx])
plt.subplot(322)
plt.plot(nypred[::y_pred.shape[1],:])
#plt.plot(nypred[::,:])
print(labelTest[idx][0])

y_pred2 = model.predict(trajTestNorm)

y_pred2.shape

idx2 = 15
plt.subplot(321)
plt.plot(x_test[idx2])
plt.subplot(322)
plt.plot(np.squeeze(y_pred2[idx2,:,:]))
print(labelTest[idx2][0])

print("trajTestNorm.shape = ",trajTestNorm.shape)
print("labelTest.shape = ", labelTest.shape)

trajdata = np.expand_dims(trajTestNorm[0,:,0],axis=0)
print("trajdata.shape = ",trajdata.shape)

labeldata = np.expand_dims(labelTest[0,:,:],axis=0)
print("labeldata.shape = ",labeldata.shape)

win = 10
#for m = in range(TrajTestNorm.shape[0]):
for m in range(500):
    trajdata = np.expand_dims(trajTestNorm[m,:,0],axis=0)
    labeldata = np.expand_dims(labelTest[m,:,:],axis=0)
    for k in range((trajdata.size)-win+1):
        if k==0:
            v = trajdata[:,k:k+win]
            w = labeldata[:,k:k+win,:]
            #print(v.shape)
        else:
            v = np.concatenate((v,trajdata[:,k:k+win]),axis=0)
            w = np.concatenate((w,labeldata[:,k:k+win,:]),axis=0)
    ve = np.expand_dims(v,axis=2)
    if m == 0:
        pred_rollup = model.predict(ve)
        real_rollup = np.copy(w)
    else:
        pred_ve = model.predict(ve)
        pred_rollup = np.concatenate((pred_rollup,pred_ve),axis=1)
        real_rollup = np.concatenate((real_rollup,w),axis=1)
        
    #pred_ve = model.predict(ve)
    
    
    #print("ve.shape = ",ve.shape)
    #print("ve[0,:,0] = ",ve[0,:,0])
    #print("pred_ve.shape = ",pred_ve.shape)
    
    
print("pred_rollup.shape = ",pred_rollup.shape)
ds_pred_rollup = pred_rollup[:,0::10,:]
print("ds_pred_rollup.shape = ",ds_pred_rollup.shape)

print("real_rollup.shape = ",real_rollup.shape)
ds_real_rollup = real_rollup[:,0::10,:]
print("ds_real_rollup.shape = ",ds_real_rollup.shape)

print(ds_real_rollup[0:10,0,:])
print(ds_pred_rollup[0:10,0,:])

for idx2 in range(30):
    plt.figure()
    plt.subplot(321)
    plt.plot(x_test[idx2])
    plt.title('label = '+str(labelTest[idx2][0]))
    plt.subplot(322)
    plt.plot(ds_pred_rollup[:,idx2,0],label='[1 0 0]')
    plt.plot(ds_pred_rollup[:,idx2,1],label='[0 1 0]')
    plt.plot(ds_pred_rollup[:,idx2,2],label='[0 0 1]')
    plt.legend()

# Blue is [1 0 0]
# Orange is [0 1 0]
# Green is [0 0 1]

