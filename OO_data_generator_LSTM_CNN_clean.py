import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Flatten, Conv1D, BatchNormalization
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class ProxyData:
    #######################################################
    #                                                     #
    # ProxyData Class:                                    #
    #                                                     #
    # This class contains the functions to support        #
    # proxy data generation associated with signatures    #
    # pertinent to Aegis BMD                              #
    #                                                     #
    #######################################################
    def __init__(self,nrecords,nsamples):
        self.nrecords = nrecords
        self.nsamples = nsamples
    
    
    def JD_Kou_Process(self,mu,lm,p,e1,e2,sig):
        ############################################################
        #                                                          #
        # JD_Kou_Process(self,mu,lm,p,e1,e2,sig)                   #
        # Member function designed to generate data                #
        # that has Gaussian components and Jump Components         #
        #                                                          #
        # Inputs:                                                  #
        #      mu = deterministic drift                            #
        #      lm = Poisson process arrival rate                   #
        #      p  = probability of an up-jump                      #
        #      e1 = parameter of up-jump (measure of magitude)     #
        #      e2 = parameter of down-jump (measure of magnitude)  #
        #      sig = Gaussian component                            #
        #      self.nsamples = num of samples                      #
        # Output:                                                  #
        #      signature_out = output signature                    #
        #                                                          #
        ############################################################
        
        ts = np.linspace(0,1000,self.nsamples+1).tolist()
        # Note that the fixed number of 1000 is arbitrary
        # since the key parameter is the number of samples, nsamples
        
        T = ts[-1]
        
        # Simulate the number of jumps
        Np = np.random.poisson(lm*T,1)
        
        L = len(ts)
        
        X = []
        Jumps_ts = np.zeros(L)
        tJumps_ts = []
        D_diff = []
        
        
        # Simulate jump arrival time
        t = T*np.random.random(Np)
        t = np.sort(t)
        
        # Simulate jump size
        ww = np.random.binomial(1,p,Np)
        d1 = np.random.exponential(e1,Np)
        d2 = np.random.exponential(e2,Np)
        S = ww*d1 - (1-ww)*d2
        
        # Put things together
        CumS = np.cumsum(S)
        for idx2 in range(L-1):
            Events = np.sum(t<=ts[idx2])
            Jumps_ts[idx2] = 0
            if Events:
                Jumps_ts[idx2] = CumS[Events-1]
                
            tJumps_ts = Jumps_ts[idx2].tolist()
            X.append(tJumps_ts)
        
        for k in range(L):
            Dt = ts[k]
            if k>0:
                Dt = ts[k] - ts[k-1]
                D_diff.append(mu*Dt + sig*np.sqrt(Dt)*np.random.randn())
        
        signature_out = np.asarray(D_diff) + np.asarray(X)
        
        return signature_out
       
    def GenerateJumpDiffusionData(self,mu,lm,p,e1,e2,sig):
        x0 = np.expand_dims(ProxyData.JD_Kou_Process(self,mu,lm,p,e1,e2,sig),axis=1)
        for k in range(self.nrecords-1):
            tmpjd = np.expand_dims(ProxyData.JD_Kou_Process(self,mu,lm,p,e1,e2,sig),axis=1)
            x0 = np.concatenate((x0,tmpjd),axis=1)
            
        return x0
      
object1 = ProxyData(100,100)
tmp0 = object1.JD_Kou_Process(0,0.1,0.5,5,5,0.1)
plt.plot(tmp0)
x0 = object1.GenerateJumpDiffusionData(0,0.1,0.5,5,5,0.1)
plt.plot(x0)
plt.show()

class ObjectMotion:
    ##################################################
    #                                                #
    # ObjectMotion Class                             #
    #                                                #
    # This class is designed to create object motion #
    # associated with global and local kinematics    #
    #                                                #
    ##################################################
    def __init__(self,nsamples):
        self.nsamples = nsamples
        
    def CalculateAspectAngle(self,a,b):
        #################################################################
        #                                                               #
        # CalculateAspectAngle(a,b):                                    #
        # Utility Function                                              #
        # Calculate the Aspect Angle of an object                       # 
        # based upon the viewing position of the Observer               #
        #                                                               #
        # Inputs:                                                       #
        #      a = Vector from Observer to Object                       #
        #          (ex. xyz of Object - xyz of Viewer)                  #
        #          Must be a 2-D array representation                   #
        #          (ex. np.array[[1,0]])                                #
        #      b = Vector containing the Object Velocity                #
        #          Must be a 2-D array represenation                    #
        #          (ex. np.array[[10,10]])                              #
        #                                                               #
        # Outputs:                                                      #
        #      AspectAngle = Aspect angle of object seen from observer  #
        #                                                               #
        #################################################################
        
        if a.shape[1] > 1:
            DotProduct = np.sum(a*b,axis=1)
            MagnitudeOne = np.linalg.norm(a,axis=1)
        else:
            Magnitude = np.linalg.norm(a)
            DotProduct = np.sum(a*b)
        MagnitudeTwo = np.linalg.norm(b)
        RawAspect = np.arccos(DotProduct/(MagnitudeOne*MagnitudeTwo)) - np.pi
        AspectAngleOut = np.abs(RawAspect)
        
        return AspectAngleOut
    
    def GenerateSpecular(self,SpecularPosition,SpecularAmplitude):
        ########################################################################################
        #                                                                                      #
        # GenerateSpecular(SpecularPosition, SpecularAmplitude)                                #
        # This function generates a specular profile of an object                              #
        # based upon the viewing position of the observer                                      #
        # relative to the local geometry of the object                                         #
        #                                                                                      #
        # Inputs:                                                                              #
        #      SpecularPosition = Locates where in the sample window to place the specular     #
        #      SpecularAmplitude = Gives the size of the specular                              #
        #                                                                                      #
        # Outputs:                                                                             #
        #      SpecularOut = return the specular signature of the object                       #                                                         #
        #                                                                                      #
        ########################################################################################
        ViewPosition = np.array([0,0])
        ObjectVelocity = np.array([1,0])
        
        # Simulated Straight-Line Trajectory
        trajectory_straight_line = np.zeros((self.nsamples,2))
        for k in range(self.nsamples):
            trajectory_straight_line[k,0] = k-SpecularPosition #N/2
            trajectory_straight_line[k,1] = 10
        
        #plt.plot(trajectory_straight_line)
        #plt.show()
        
        object_range = trajectory_straight_line - ViewPosition
        aspectangle = ObjectMotion.CalculateAspectAngle(self,object_range,ObjectVelocity)
        
        angleidx = aspectangle*180/np.pi > np.pi/2
        aspectangle[angleidx] = np.pi/2 - np.abs(aspectangle[angleidx]-np.pi/2)
        
        SignatureOut = aspectangle*SpecularAmplitude
        
        #plt.plot(SignatureOut)
        
        return SignatureOut
    
    def GenerateTumble(self,ObjectRange,RotationRate,ReferenceLevel):
        ###############################################################################
        #                                                                             #
        # GenerateTumble(ObjectRange,RotationRate,ReferenceLevel)                     #
        # This function generates a tumble sequence profile of the object             #
        # based upon the viewing position of the observer                             #
        # relative to the local geometry of the object                                #
        #                                                                             #
        # Inputs:                                                                     #
        #      ObjectRange = Range of Object Relative to Viewer                       #
        #      RotationRate = Rotation Rate of Object                                 #
        #      ReferenceLevel = Zero Level of Tumble Amplitude                        #
        #                                                                             #
        # Outputs:                                                                    #
        #      TumbleOut = return the tumble signature of the object                  #
        #                                                                             #
        ###############################################################################
        
        InitialObjectOrientation = np.array([1,0])
        TimeStep = np.linspace(0,100,self.nsamples)
        ObjectOrientation = np.zeros((N,2))
        
        for k in range(self.nsamples):
            ObjectOrientation[k,0] = np.cos(RotationRate*TimeStep[k])*InitialObjectOrientation[0] \
            - np.sin(RotationRate*TimeStep[k])*InitialObjectOrientation[1]
            ObjectOrientation[k,1] = np.sin(RotationRate*TimeStep[k])*InitialObjectOrientation[0] \
            + np.cos(RotationRate*TimeStep[k])*InitialObjectOrientation[1]
            
        TumbleOut = ReferenceLevel*ObjectMotion.CalculateAspectAngle(self,np.tile(ObjectRange,(self.nsamples,1)),ObjectOrientation)
        
        return TumbleOut
      
nrecs = 5000
N = 500
object_motion = ObjectMotion(N)
ViewPosition = np.array([[0,0]])
ObjectPosition = np.array([[11,10]])
ObjectVelocity = np.array([[1,0]])
ObjectRange = ObjectPosition - ViewPosition
aspectangle = object_motion.CalculateAspectAngle(ObjectRange,ObjectVelocity)
print(aspectangle)

signature_data_specular = np.zeros((N,nrecs))

object_motion.GenerateSpecular(np.random.randint(50,450),np.abs(np.random.randn()))

for k in range(nrecs):
    signature_data_specular[:,k] = object_motion.GenerateSpecular(np.random.randint(5,N-5),np.abs(np.random.randn()))
    
plt.plot(signature_data_specular)
plt.show()


signature_data_tumble_high = np.zeros((N,nrecs))
for k in range(nrecs):
    tmpsignature = np.random.randint(1,5,1)*object_motion.GenerateTumble(ObjectRange,np.random.randn(),np.random.randn())
    signature_data_tumble_high[:,k] = np.abs(tmpsignature) - np.mean(np.abs(tmpsignature))
    

for k in range(nrecs):
    left_side_random = int(np.random.randint(10,100,1))
    right_side_random = int(np.random.randint(10,100,1))
    signature_data_tumble_high[0:left_side_random,k]= 0
    signature_data_tumble_high[-right_side_random:-1,k] = 0

print(signature_data_tumble_high.shape)
plt.plot(signature_data_tumble_high[:,0:6])
#plt.plot(signature_data_tumble_high)
#plt.show()


# Generate data for all types
DATATYPE = 1

if DATATYPE == 0:
    data_levels = np.array([0.02,0.01,0.005,0.001])
    # Create Brownian Motion
    x0 = np.random.randn(nrecs,N)*data_levels[0]
    x1 = np.random.randn(nrecs,N)*data_levels[1]
    x2 = np.random.randn(nrecs,N)*data_levels[2]
    x3 = np.random.randn(nrecs,N)*data_levels[3]

else:
    ProxyDataObject1 = ProxyData(nrecs,N)
    
    # Generate High Frequency Tumbles
    signature_data_tumble_high = np.zeros((nrecs,N))
    for k in range(nrecs):
        tmpsignature = np.random.randint(1,5,1)*object_motion.GenerateTumble(ObjectRange,np.random.randn(),np.random.randn())
        signature_data_tumble_high[k,:] = np.abs(tmpsignature) - np.mean(np.abs(tmpsignature))
        
    for k in range(nrecs):
        left_side_random = int(np.random.randint(10,100,1))
        right_side_random = int(np.random.randint(10,100,1))
        signature_data_tumble_high[k,0:left_side_random]= 0
        signature_data_tumble_high[k,-right_side_random:-1] = 0
    
    plt.figure()
    plt.plot(signature_data_tumble_high[0:6,:].T)

        
    # Generate low frequency tumbles
    signature_data_tumble_low = np.zeros((nrecs,N))
    for k in range(nrecs):
        tmpsignature = np.random.randint(1,5,1)*object_motion.GenerateTumble(ObjectRange,np.random.randn()/10,np.random.randn())
        # Scale back to zero mean
        signature_data_tumble_low[k,:] = np.abs(tmpsignature)-np.mean(np.abs(tmpsignature))
    
    print(signature_data_tumble_low.shape)



    
    # Configure data for label 0
    data_levels_x0 = np.array([0.05,0.4,0.1])
    x0 = np.expand_dims(ProxyDataObject1.JD_Kou_Process(0,0.0,0.5,0.5,5,0.5),axis=1)  # Gaussian Process Only
    for k in range(nrecs-1):
        tmp1jd = np.expand_dims(ProxyDataObject1.JD_Kou_Process(0,0.0,0.5,0.5,5,0.5),axis=1)  # Gaussian Process Only
        x0 = np.concatenate((x0,tmp1jd),axis=1)
    
    print('xo.shape = ', x0.shape)
    print('signature_data_specular.shape = ', signature_data_specular.shape)
    
    x0 = x0.T*data_levels_x0[0] + signature_data_specular.T*data_levels_x0[1] + signature_data_tumble_low*data_levels_x0[2]
    
    # Configure data for Label 1
    data_levels_x1 = np.array([0.02])
    x1 = np.expand_dims(ProxyDataObject1.JD_Kou_Process(0,0.0005,0.5,6,6,2),axis=1)
    for k in range(nrecs-1):
        tmp1jd = np.expand_dims(ProxyDataObject1.JD_Kou_Process(0,0.0005,0.5,6,6,2),axis=1)
        x1 = np.concatenate((x1,tmp1jd),axis=1)
        
    x1 = x1.T*data_levels_x1[0]
    
    # Configure data for Label 2
    data_levels_x2 = np.array([0.1])
    x2 = np.random.randn(nrecs,N)*data_levels_x2[0]
    
    # Configure data for Label 3
    data_levels_x3 = np.array([0.01,0.3])
    x3 = np.random.randn(nrecs,N)*data_levels_x3[0] + signature_data_tumble_high*data_levels_x3[1]
    
    print(x0.shape)
    print(x1.shape)
    
    v0 = np.copy(x0)
    v1 = np.copy(x1)
    v2 = np.copy(x2)
    v3 = np.copy(x3)
    
    # Create Drift Components
    b0 = b1 = b2 = b3 = 0
    
    p0 = v0 + b0
    p1 = v1 + b1
    p2 = v2 + b2
    p3 = v3 + b3
    
    print(p0.shape)
    
    
    #if FILTER == 1:
        
    
    # Artificially creating sequences that are of different non-zero values
    # The rest of the sequences are zero padding to the maximum length of the
    # array
    
    for k in range(nrecs):
        nzlen0 = np.random.randint(10,40)
        p0[k,-nzlen0:] = 0
        nzlen1 = np.random.randint(10,40)
        p1[k,-nzlen1:] = 0
        nzlen2 = np.random.randint(10,40)
        p2[k,-nzlen2:] = 0
        nzlen3 = np.random.randint(10,40)
        p3[k,-nzlen3:] = 0
        
        
if (1):
    plt.figure()
    plt.subplot(141), plt.plot(p0.T)
    plt.title('class 1')
    plt.subplot(142), plt.plot(p1.T)
    plt.title('class 2')
    plt.subplot(143), plt.plot(p2.T)
    plt.title('class 3')
    plt.subplot(144), plt.plot(p3.T)
    plt.title('class 4')
    plt.show()
    
    plt.figure()
    plt.plot(p0.T,'g')
    plt.plot(p1.T,'r')
    plt.plot(p2.T,'b')
    plt.plot(p3.T,'m')
    
    
#h0 = np.tile(np.array([1.,1.,1.,1.,1.])/5,(p0.shape[0],1))
#h0 = np.ones(10)/10
#h0 = np.array([-1.,0.,0.,0.,2.,0.,0.,0.,-1])
h0 = np.array([1., 2., 4., 2., 1.])/10
print(h0.shape)
cp0 = np.apply_along_axis(lambda m: np.convolve(m, h0,mode='same'),axis=1,arr=p0)
cp1 = np.apply_along_axis(lambda m: np.convolve(m, h0,mode='same'),axis=1,arr=p1)
cp2 = np.apply_along_axis(lambda m: np.convolve(m, h0,mode='same'),axis=1,arr=p2)
cp3 = np.apply_along_axis(lambda m: np.convolve(m, h0,mode='same'),axis=1,arr=p3)


#cp0.shape
#plt.figure()
#plt.subplot(211)
#plt.plot(cp0[0,:])
#plt.subplot(212)
#plt.plot(p0[0,:])
#plt.show()
#cp1 = np.convolve(p1,h0,'same')
#cp2 = np.convolve(p2,h0,'same')
#cp3 = np.convolve(p3,h0,'same')

PLOTRESULTS = 0

if PLOTRESULTS == 1:
    plt.figure()
    plt.subplot(141), plt.plot(cp0.T)
    plt.title('class 1')
    plt.subplot(142), plt.plot(cp1.T)
    plt.title('class 2')
    plt.subplot(143), plt.plot(cp2.T)
    plt.title('class 3')
    plt.subplot(144), plt.plot(cp3.T)
    plt.title('class 4')
    plt.show()
    
    plt.figure()
    plt.plot(cp0.T,'g')
    plt.plot(cp1.T,'r')
    plt.plot(cp2.T,'b')
    plt.plot(cp3.T,'m')
    
FILTER = 0

if FILTER == 1:
    p0 = cp0
    p1 = cp1
    p2 = cp2
    p3 = cp3

print("maximum value of data = ", np.max([p0, p1, p2, p3]))
print("minimum value of data = ", np.min([p0, p1, p2, p3]))

tX0 = np.concatenate((p0,p1), axis=0)
tX1 = np.concatenate((tX0,p2), axis=0)
X = np.concatenate((tX1,p3), axis=0)

print("X shape = ", X.shape)
print("minimum value of data = ", np.min(X))
print("maximum value of data = ", np.max(X))
print("mean value of data = ", np.mean(X))
print("Scaling data by min and max")

rsX = 2*(X-np.min(X))/(np.max(X)-np.min(X)) - 1
print("scaled minimum of data = ", np.min(rsX))
print("scaled maximum of data = ", np.max(rsX))


ncategories = 4
y0 = np.zeros((nrecs,N))
y1 = np.ones((nrecs,N))
y2 = np.ones((nrecs,N))*2
y3 = np.ones((nrecs,N))*3

tY0 = np.concatenate((y0,y1),axis=0)
tY1 = np.concatenate((tY0,y2),axis=0)
Y = np.concatenate((tY1,y3),axis=0)

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

wsize = 10
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


plt.plot(dsbatch_train[0,:,:])


class ClassifierModel:
    #######################################################
    #                                                     #
    # ClassifierModel Class:                              #
    #                                                     #
    # This class contains the classifier models           #
    # to support analysis of the data                     #
    #                                                     #
    #######################################################
    def __init__(self,modeltype,ncategories):
        self.modeltype = modeltype
        self.ncategories = ncategories
        
    def BuildModelLSTM(self,nlayers,nodesperlayer):
        inputs = tf.keras.Input(name='inputs',shape=(None,1))
        layer = LSTM(nodesperlayer,activation='tanh',return_sequences=True)(inputs)
        
        # Loop over the number of additional layers specified 
        # in the member function
        for k in range(nlayers):
            layer = LSTM(nodesperlayer,activation='tanh',return_sequences=True)(layer)
            
        layer = Dense(self.ncategories,activation='softmax')(layer)
        model = tf.keras.Model(inputs=inputs,outputs=layer)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    def BuildModelResNetLSTM1(self,nodesperlayer,dropout):
        inputs = tf.keras.Input(name='inputs',shape=(None,1))
        a = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(inputs)
        x = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(a)
        a = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(a)
        x = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(x)
        x = LSTM(nodesperlayer,activation='tanh',dropout=0.2,recurrent_dropout=0.0,return_sequences=True)(x)
        b = tf.keras.layers.Add()([a,x])
        
        # Convolution layer as the last layer
        x = tf.keras.layers.Conv1D(filters=15,kernel_size =15,padding='same',activation='relu')(b)
        
        layer = Dense(self.ncategories,activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs,outputs=layer)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    def BuildModelResNetLSTM2(self,nodesperlayer,dropout):
        inputs = tf.keras.Input(name='inputs',shape=(None,1))
        a = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(inputs)
        x = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(a)
        a = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(a)
        x = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(x)
        x = LSTM(nodesperlayer,activation='tanh',dropout=dropout,recurrent_dropout=0.0,return_sequences=True)(x)
        b = tf.keras.layers.Add()([a,x])
        
        # LSTM Layer as the last layer
        x = LSTM(nodesperlayer,activation='tanh',dropout=dropout,recurrent_dropout=0.0,return_sequences=True)(b)
        
        layer = Dense(self.ncategories,activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs,outputs=layer)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    
    def BuildModelResNetLSTM3(self,nodesperlayer,dropout):
        inputs = tf.keras.Input(name='inputs',shape=(None,1))
        a = LSTM(nodesperlayer,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(inputs)
        x = LSTM(nodesperlayer-5,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(a)
        a = LSTM(nodesperlayer-10,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(a)
        x = LSTM(nodesperlayer-5,activation='tanh',dropout=0.0,recurrent_dropout=0.0,return_sequences=True)(x)
        x = LSTM(nodesperlayer,activation='tanh',dropout=dropout,recurrent_dropout=0.0,return_sequences=True)(x)
        b = tf.keras.layers.Add()([a,x])
        
        # LSTM Layer as the last layer
        x = LSTM(nodesperlayer,activation='tanh',dropout=dropout,recurrent_dropout=0.0,return_sequences=True)(b)
        
        layer = Dense(self.ncategories,activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs,outputs=layer)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    
    def BuildResNet1D(self):
        
        inputs = tf.keras.Input(name='inputs',shape=(None,1))
        #inputs = tf.keras.Input(shape=(200,1), name='inputs')
      
        Pre1 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal', name='pre1_conv')(inputs)
        Pre1 = BatchNormalization(axis=1, name='pre1_bn')(Pre1)
        Pre1 = Activation('relu', name='pre1_relu')(Pre1)

        Res1 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(Pre1)
        Res1 = BatchNormalization(axis=1)(Res1)
        Res1 = Dropout(0.25)(Res1)
        Res1 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(Res1)

        Skip1 = Conv1D(16, 1)(Pre1)

        start_out = keras.layers.add([Res1, Skip1])


        mid1 = BatchNormalization(axis=1)(start_out)
        mid1 = Activation('relu')(mid1)
        mid1 = Dropout(0.25)(mid1)
        mid1 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(mid1)
        mid1 = BatchNormalization(axis=1)(mid1)
        mid1 = Activation('relu')(mid1)
        mid1 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(mid1)

        mid_skip1 = Conv1D(16, 1)(start_out)

        mid1_out = keras.layers.add([mid1, mid_skip1])


        mid2 = BatchNormalization(axis=1)(mid1_out)
        mid2 = Activation('relu')(mid2)
        mid2 = Dropout(0.25)(mid2)
        mid2 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(mid2)
        mid2 = BatchNormalization(axis=1)(mid2)
        mid2 = Activation('relu')(mid2)
        mid2 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(mid2)

        mid_skip2 = Conv1D(16, 1)(mid1_out)

        mid2_out = keras.layers.add([mid2, mid_skip2])


        mid3 = BatchNormalization(axis=1)(mid2_out)
        mid3 = Activation('relu')(mid3)
        mid3 = Dropout(0.25)(mid3)
        mid3 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(mid3)
        mid3 = BatchNormalization(axis=1)(mid3)
        mid3 = Activation('relu')(mid3)
        mid3 = Conv1D(16, 5, padding='same', kernel_initializer='he_normal')(mid3)
        mid3 = MaxPooling1D(pool_size=3)(mid3)

        mid_skip3 = Conv1D(16, 1)(mid3_out)
        mid_skip3 = MaxPooling1D(pool_size=3)(mid_skip3)

        mid3_out = keras.layers.add([mid3, mid_skip3])


        end = BatchNormalization(axis=1)(mid3_out)
        end = Activation('relu')(end)
        end = Flatten()(end)
        end = Dense(200)(end)
        end = BatchNormalization(axis=1)(end)
        end = Activation('relu')(end)
        end = Dropout(0.25)(end)
        end = Dense(5)(end) # Number of classes
        outputs = Activation('softmax')(end)

        model = tf.keras.Model(inputs=inputs,outputs=layer)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    def BuildCNN1DModel1(self):
                         
        inputs = Input(name='inputs',shape=(None,1))
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        #x = GlobalMaxPooling1D()(x)
        outputs = Dense(self.ncategories,activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs,outputs=outputs)
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    def BuildCNN1DModel2(self):
                         
        inputs = Input(name='inputs',shape=(None,1))
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(self.ncategories,activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs,outputs=outputs)
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
    
    def BuildCNN1DModel3(self):    
        inputs = Input(name='inputs',shape=(None,1))
        x = Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2,strides=1,padding='same')(x)
        x = Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(x)
        x = MaxPooling1D(pool_size=2,strides=1,padding='same')(x)
        x = Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(x)
        x = MaxPooling1D(pool_size=2,strides=1,padding='same')(x)
        x = Conv1D(filters=25,kernel_size = 10,padding='same',activation='relu')(x)
        x = MaxPooling1D(pool_size=2,strides=1,padding='same')(x)
        outputs = Dense(self.ncategories,activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs,outputs=outputs)
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return model
        
        
    def FitModel(self,model,dsbatch_train,dslabel_train,nepochs,validation_split):
        print("Training the model"+str(model))
        history = model.fit(dsbatch_train, dslabel_train, epochs=nepochs, validation_split = validation_split)
        
        return history
      

classifier1 = ClassifierModel(1,4)
classifier2 = ClassifierModel(1,4)

cnn1dmodel1 = classifier1.BuildCNN1DModel1()
cnn1dmodel2 = classifier1.BuildCNN1DModel2()

print(dsbatch_train.shape)
print(dslabel_train.shape)

###################
# Train the Model #
###################
print('Training the model')                     

history1 = classifier1.FitModel(cnn1dmodel1,dsbatch_train,dslabel_train,400,0.3)

# For this model of the 1D CNN, the labels are such that 1 label is assigned per sequence
# rather than assigning a label PER sample in each sequence
one_label_per_sequence = np.squeeze(dslabel_train[:,0,:])
print('dsbatch_train.shape = ',dsbatch_train.shape)
print('Revised dslabel_train.shape = ',one_label_per_sequence.shape)
#history2 = classifier1.FitModel(cnn1dmodel1,dsbatch_train,np.squeeze(dslabel_train[:,0,:]),400,0.3)
history2 = classifier1.FitModel(cnn1dmodel1,dsbatch_train,one_label_per_sequence,400,0.3)

#, validation_data=(trajTestNorm, labelTest))
dsbatch_train.shape

plt.figure()
plt.subplot(211)
plt.plot(history1.history['loss'],'b',label='training loss')
plt.plot(history1.history['val_loss'],'g',label='validation_loss')
plt.legend()
plt.subplot(212)
plt.plot(history1.history['accuracy'],'b',label='training_accuracy')
plt.plot(history1.history['val_accuracy'],'g',label='validation_accuracy')
plt.legend()
history1.history.keys()


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
  
y_pred2 = cnn1dmodel1.predict(trajTestNorm)
trajTestNorm.shape

np.expand_dims(trajTestNorm[m,:,0],axis=0).shape
np.expand_dims(labelTest[m,:,:],axis=0).shape

trajdata.size

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
    
ve.shape

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
        #pred_rollup = ResNetLSTM2model1.predict(ve)
        pred_rollup = cnn1dmodel1.predict(ve)
        real_rollup = np.copy(w)
    else:
        #pred_ve = ResNetLSTM2model1.predict(ve)
        pred_ve = cnn1dmodel1.predict(ve)
        pred_rollup = np.concatenate((pred_rollup,pred_ve),axis=1)
        real_rollup = np.concatenate((real_rollup,w),axis=1)
        
print("pred_rollup.shape = ",pred_rollup.shape)
ds_pred_rollup = pred_rollup[:,0::10,:]
print("ds_pred_rollup.shape = ",ds_pred_rollup.shape)

print("real_rollup.shape = ",real_rollup.shape)
ds_real_rollup = real_rollup[:,0::10,:]
print("ds_real_rollup.shape = ",ds_real_rollup.shape)

subplot(211)
plt.plot(pred_rollup[:,2,:])
plt.show()
subplot(2)
plt.plot(real_rollup[:,2,:])


for idx2 in range(100):
    plt.figure()
    plt.subplot(321)
    plt.plot(x_test[idx2])
    plt.title('label = '+str(labelTest[idx2][0]))
    plt.subplot(322)
    #plt.plot(ds_pred_rollup[:,idx2,0],label='[1 0 0]')
    #plt.plot(ds_pred_rollup[:,idx2,1],label='[0 1 0]')
    #plt.plot(ds_pred_rollup[:,idx2,2],label='[0 0 1]')
    plt.plot(ds_pred_rollup[:,idx2,0],'r')
    plt.plot(ds_pred_rollup[:,idx2,1],'b')
    plt.plot(ds_pred_rollup[:,idx2,2],'g')
    plt.plot(ds_pred_rollup[:,idx2,3],'c')
    

