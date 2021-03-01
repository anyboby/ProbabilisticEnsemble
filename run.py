from pe_factory import build_PE
from utils import average_dkl, median_dkl
import numpy as np
import tensorflow as tf


x = np.random.normal(0, scale=30, size=20000)[...,None]
y = np.cos(x) * (np.sin(x) + 3 * np.cos(x)**2) * np.exp(np.cos(x/5))
x_dims = np.shape(x)[-1]
y_dims = np.shape(y)[-1]
loss = 'MSPE'
hidden_dims = (30,30)
num_nets = 7
num_elites = 5
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = build_PE(in_dim=x_dims, 
                        out_dim=y_dims,
                        name='PE',
                        loss=loss,
                        hidden_dims=hidden_dims,
                        lr=1e-3,
                        num_networks=num_nets, 
                        num_elites=num_elites,
                        use_scaler_in = True,
                        use_scaler_out = True,
                        decay=1e-5,
                        max_logvar=.5,
                        min_logvar=-10,
                        session=sess)

model_metrics = model.train(x, 
                            y,
                            batch_size=64, #512
                            max_epochs=50000, # max_epochs 
                            min_epoch_before_break=500, # min_epochs, 
                            holdout_ratio=0.2, 
                            max_t=5000
                                )
