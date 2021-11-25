import tensorflow as tf
from tensorflow import keras
import numpy as np

x_in = np.array([[1,2,3],[4,5,6],[7,8,9],[2,4,5]])
x_in_reshape = x_in.reshape((1,x_in.shape[0],x_in.shape[1], 1))
x_in_reshape.shape

kernel_in = np.array([[1,2],[3,4]])
kernel_in_reshape = kernel_in.reshape((kernel_in.shape[0],kernel_in.shape[1],1,1))
kernel_in_reshape.shape

# Needed to reshape both x_in and kernel_in to match the format for the convolution 
# which states that the input tensor needs to follow: [batch,input_height,input_width,input_channel] = [1,m,n,1] 
# and the input kernel needs to follow: [filter_width,filter_height,input_channel,output_channel] = [l,k,1,1]

x0 = tf.constant(x_in_reshape, dtype=tf.float32)
kernel0 = tf.constant(kernel_in_reshape, dtype=tf.float32)
tf.nn.conv2d(x0,kernel0,strides=[1,1,1,1],padding='VALID')

# Check
from scipy import signal
a0 = signal.convolve2d(kernel_in,x_in,mode='valid')
print(a0)

print(x_in)
print(kernel_in)

print("First Row")
m = 0
print(x_in[m,0]*  kernel_in[1,1]+x_in[m,1]*  kernel_in[1,0]+ \
      x_in[m+1,0]*kernel_in[0,1]+x_in[m+1,1]*kernel_in[0,0])

print(x_in[m,1]*  kernel_in[1,1]+x_in[m,2]*  kernel_in[1,0]+ \
      x_in[m+1,1]*kernel_in[0,1]+x_in[m+1,2]*kernel_in[0,0])

print("Second Row")
m = 1
print(x_in[m,0]*  kernel_in[1,1]+x_in[m,1]*  kernel_in[1,0]+ \
      x_in[m+1,0]*kernel_in[0,1]+x_in[m+1,1]*kernel_in[0,0])

print(x_in[m,1]*  kernel_in[1,1]+x_in[m,2]*  kernel_in[1,0]+ \
      x_in[m+1,1]*kernel_in[0,1]+x_in[m+1,2]*kernel_in[0,0])

print("Third Row")
m = 2
print(x_in[m,0]*  kernel_in[1,1]+x_in[m,1]*  kernel_in[1,0]+ \
      x_in[m+1,0]*kernel_in[0,1]+x_in[m+1,1]*kernel_in[0,0])
      
      
print(kernel_in)
print(kernel_in.T)
print(np.matmul(kernel_in.T,np.array([[0,1],[0,0]])))
 
angle = np.pi/2;
rot = np.array([[np.cos(angle),np.sin(angle)],[np.sin(angle),np.cos(angle)]])
print(np.matmul(kernel_in,rot))

print(np.matmul(kernel_in,rot).T)
print(np.matmul(np.matmul(kernel_in,rot).T,rot).T)

print("The goal = ")
print(np.array([[4,3],[2,1]]))


print(x_in[m,1]*  kernel_in[1,1]+x_in[m,2]*  kernel_in[1,0]+ \
      x_in[m+1,1]*kernel_in[0,1]+x_in[m+1,2]*kernel_in[0,0])
