import numpy as np
import matplotlib.pyplot as plt
from filter1 import rng_2s_orse 

n = 200
T = 0.25

srr = 0.4
Au = 0.08

phase = np.asarray(range(0,n,1))

xta = 10+10*np.sin(phase/20.) + 1*np.random.randn(n)

xva = np.ones(n)*(xta[n-1]-xta[0])/((n-1)*T)

xma = xta + np.random.randn(n)*srr*1

#vo = np.array([ [0],[0],[0] ])
vo = np.zeros(3).reshape(-1,1)

xs = np.zeros(4).reshape(-1,1)

alphaLtrue = np.zeros(n)
aobjsmth = alphaLtrue
rngdwella = np.ones(n)

time = np.zeros(n)
na = np.zeros(n)
rng_2s_states = np.zeros((2,1))
rng_2s_statesa = np.zeros((n,2))
rng_2s_Sa = np.zeros((n,2,2))
rng_2s_Ma = np.zeros((n,2,2))
rng_2s_DLDa = np.zeros((n,2,2))




for k in range(n):
  
  time[k] = k*T
  na[k] = k

  # temp
  xm = xma[k]

  # 1st and 2nd time filter init

  # 1st time filter init
  if k == 0:

    # states
    xs[0] = xm
    xs[1:] = vo 

    xp = xs

    rng_2s_states[0][0] = np.linalg.norm(xm)
    rng_2s_states[1][0] = 0

    rng_2s_statesa[0][0] = rng_2s_states[0][0]
    rng_2s_statesa[0][1] = rng_2s_states[1][0]  

    rng_2s_Sa[0][:][:] = np.zeros((2,2)) 

  # 2nd time filter init
  elif k == 1:

    # states
    rng_2s_states[0][0] = np.linalg.norm(xm)
    rng_2s_states[1][0] = ( np.linalg.norm(xm) - rng_2s_states[0][0] + 0.5 * aobjsmth[0] * T**2 ) / T

    # covariances
    rng_2s_S = np.array([ [srr**2, srr**2/T],[srr**2/T, 2*srr**2/T**2] ])

    rng_2s_D = np.array([ [0],[0] ])

    rng_2s_M = rng_2s_S

    rng_2s_statesa[1][0] = rng_2s_states[0][0]
    rng_2s_statesa[1][1] = rng_2s_states[1][0]

    rng_2s_Sa[1][:][:] = rng_2s_S

  # Normal Processing
  else:
    alpha2sors = aobjsmth[k]

    MM = srr**2

    rng_2s_states, rng_2s_M, rng_2s_D, rng_2s_S, K = rng_2s_orse(
	np.linalg.norm(xm), 
	rng_2s_states, 
	rng_2s_M, 
	rng_2s_D, 
	rng_2s_S, 
	MM, 
	alpha2sors, 
	T, 
	Au, 
	rngdwella[k])

    rng_2s_statesa[k][0] = rng_2s_states[0][0]
    rng_2s_statesa[k][1] = rng_2s_states[1][0]

    rng_2s_Sa[k][:][:] = rng_2s_S

    rng_2s_Ma[k][:][:] = rng_2s_M

    rng_2s_DLDa[k][:][:] = np.matmul(rng_2s_D,Au**2*rng_2s_D.T)  # Au is a constant



tau = (3*T/2)**0.2 * (srr/Au)**0.4

kx = (3*T/(2*tau))**0.5
sigx = kx*srr

kv = (T/tau**3)**0.5
sigv = kv*srr

Bx = Au*0.5*(T+tau)**2
Bv = Au * (T+tau)

totflterrx = (sigx**2 + Bx**2)**0.5
totflterrv = (sigv**2 + Bv**2)**0.5

# 20 Point Filter
npts = n
Spx = srr * 2/npts**0.5
Spv = srr * (12/npts**3)**0.5/T

plt.figure()
plt.plot(time, 3*(rng_2s_Sa[:,0,0].T)**0.5, 'r.')
plt.plot(time, 3*(rng_2s_Ma[:,0,0].T)**0.5, 'b.')
plt.plot(time, 3*(rng_2s_DLDa[:,0,0].T)**0.5, 'g.')

plt.xlabel('time')
plt.ylabel('3sig:position, m')
plt.title('POS: rng 2 state orse')

plt.figure()
plt.plot(time, 3*(rng_2s_Sa[:,1,1].T)**0.5, 'r.')
plt.plot(time, 3*(rng_2s_Ma[:,1,1].T)**0.5, 'b.')
plt.plot(time, 3*(rng_2s_DLDa[:,1,1].T)**0.5, 'g.')

plt.xlabel('time')
plt.ylabel('3sig:velocity, m')
plt.title('VEL: rng 2 state orse')


plt.figure()
plt.plot(time,rng_2s_statesa[:,0],'b')
plt.plot(time,xta,'g')

plt.show()
