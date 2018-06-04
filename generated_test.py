
import h5py
import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.stats import binom
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Lambda


from keras.optimizers import RMSprop,adam
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint

# partial parsing function for getting y values
def fix_end_seq(y):
    n = 0
    for x in range(len(y)):
        if y[x][0] == 0:
            n = x + 1
            break
    n = n or len(y) + 1
    # a = zip(range(1, len(y[:n])), [0.] * n)
    a = [[i, 0.] for i in range(1, n)]
    b = y[(n-1):]
    return list(a) + b

# parsing funciton for getting y values
def get_y(x, sample_len):
    #assert len(x) % sample_len == 0
    x_rev = list(x)[:]
    x_rev.reverse()
    y = []
    event_seen = False
    tte = 0
    for i in x_rev:
        if i == 1:
            tte += 1
            y.append([tte, event_seen * 1.])
            tte = -1
            event_seen = True
        else:
            tte += 1
            y.append((tte, event_seen * 1.))
    n = len(x) - sample_len
    sample = y[-sample_len:][:]
    if sample[0][0] is not 0:
        sample = fix_end_seq(sample)
    sample.reverse()
    y_samples = [sample]
    for i in range(1, n+1):
        sample = y[-(i + sample_len):-i][:]
        if sample[0][0] is not 0:
            sample = fix_end_seq(sample)
        sample.reverse()
        y_samples.append(sample)
    return y_samples

# modifying generated test set

df_gen = pd.read_csv('../../data/generated_data.csv',header=0,skiprows=1,engine='python')

df_gen.set_index('date', inplace=True)

df_gen.index = pd.DatetimeIndex(df_gen.index)
df_gen = df_gen.reindex(index = pd.date_range('07-01-2010','02-11-2105'), fill_value=0)

df_gen = df_gen.amount.apply(lambda x: 1. if x>0 else 0.)

# len 34559 80% train = 27647, 20% test = 6912

test_gen = df_gen[0:6912]

test_gen = test_gen.reset_index()
test_gen = [value[1] for value in test_gen.values]

train_gen = df_gen.values
x_gen = train_gen[:27647]

x_gen = list(x_gen)

sample_len = 42
n = len(x_gen) - sample_len
xg_train = []
for i in range(0, n + 1):
    sample = x_gen[i:(i + sample_len)]
    xg_train.append([[k] for k in sample])

xg_train = np.array(xg_train)
yg_train = np.array(get_y(x_gen, sample_len))

xg_t, yg_t = test_gen[:3456], test_gen[:3456]

xg_t = list(xg_t)

n = len(xg_t) - sample_len
xg_test = []
for i in range(0, n + 1):
    sample = xg_t[i:(i + sample_len)]
    xg_test.append([[k] for k in sample])

xg_test = np.array(xg_test)
yg_test = np.array(get_y(xg_t,sample_len))

print('x:',xg_train.shape, xg_test.shape)
print('y:',yg_train.shape, yg_test.shape)

tte_mean_train = np.nanmean(yg_train[:,:,0])
init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
mean_u = np.nanmean(yg_train[:,:,1])
init_alpha = init_alpha/mean_u
print('init_alpha: ',init_alpha,'mean uncensored: ',mean_u)

history = History()
weightwatcher = WeightWatcher()

n_features = 1

# Start building the model
model = Sequential()

model.add(GRU(1,input_shape=(xg_train.shape[1:]),activation='tanh',return_sequences=True))

model.add(Dense(2))
model.add(Lambda(wtte.output_lambda,
                 arguments={"init_alpha":init_alpha,
                            "max_beta_value":4.0}))
loss = wtte.loss(kind='discrete').loss_function

#model.load_weights('load_weight.hdf5')

model.compile(loss=loss, optimizer=adam(lr=.01))

# checkpoints

filepath = 'gen_cp/{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]
model.summary()


model.fit(xg_train,yg_train,
          epochs=60,
          batch_size=xg_train.shape[0]//10,
          verbose=1,
          validation_data=(xg_test, yg_test),
#           sample_weight = sample_weights # If varying length
          callbacks=[checkpoint, history, weightwatcher])

# predict
predicted = model.predict(xg_test, batch_size=xg_train.shape[0]//10, verbose = 1)


# granular view
a_predicted = predicted[-42:]

# plot alpha and beta values
plt.imshow(a_predicted[:,:,0],interpolation="none",cmap='binary',aspect='auto')
plt.title('alpha')
plt.colorbar(orientation="horizontal")
plt.show()
plt.imshow(a_predicted[:,:,1],interpolation="none",cmap='binary',aspect='auto')
plt.title('beta')
plt.colorbar(orientation="horizontal")
plt.show()

drawstyle = 'steps-post'

print('numpy array of predictions:',predicted)

n_timesteps = 42


alpha = predicted[:,:,0]
beta = predicted[:,:,1]

print('alpha:',alpha[0])
print('beta:',beta[0])
a = alpha[0,-1]
b = beta[0,-1]


prob_on = []
prob_within = [0]
timesteps = []
for i in range(n_timesteps):
    timesteps.append(i)
    prob_on.append(weibull.pmf(i,a,b))
    prob_within.append(weibull.cmf(i,a,b))

print('List of probabilties of occurring on index+1 day:',prob_on)
print('List of probabilities of occurring within index+1 days:',prob_within)

# plotting probability of event occurring within 4 steps; for visuals
timesteps = timesteps[0:4]
prob_within = prob_within[0:4]
plt.plot(timesteps, prob_within, color='grey')
plt.xticks(timesteps)
plt.xlabel('timesteps')
plt.ylabel('probability on occurring by timestep')

plt.show()

