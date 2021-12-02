import numpy as np
from numpy.linalg import inv

def rng_2s_orse(xm,xs,M,D,S,MM,alpha,T,Au,smoothflg):

  I2 = np.eye(2)

  F = np.array([ [1.,T],[0.,1.] ])

  G = np.array([ [T*T/2],[T] ])

  H = np.array([ [1.,0.] ])

  
  # compute Lambda matrix
  lamdorse = Au**2

  # predict states
  gp = np.array([ [0.5*alpha*T*T],[alpha*T] ])

  xp = F*xs + gp

  # predict covariance
  Mp = F*M*F.T
  Dp = F*D + G
  Sp = Mp + Dp*lambdaorse*Dp.T

  # compute gains and filter

  if smoothflg:
  
    # smooth covariance
    Q = H*Sp*H.T + MM
    K = Sp*H.T*inv(Q)
    L = I2 - K*H
    M = L*Mp*L.T + K*MM*K.T
    D = L*Dp;
    S = M + D*lambdaorse*D.T

    # smooth states
    xs = xp + K*(Xm - H*xp)
  
  else:

    # coast covariance
    M = Mp
    D = Dp
    S = Sp

    # coast states
    xs = xp



  if 0:
    S
    D*lambdaorse*D.T
    M**0.5
    K.T



