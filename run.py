from pe_factory import build_PE
from utils import average_dkl, median_dkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


COLOR = ['red', 'blue', 'orange', 'purple', 'turquoise']
training_bs = 5000
x = np.random.normal(0.7, scale=3.7, size=training_bs)[...,None]
y =  100 * (x + 5*np.cos(x) + (np.tanh(x)+1.5) * np.random.normal(0, scale=1, size=training_bs)[...,None]) # * (np.sin(x) + 3 * np.cos(x)**2)
x_dims = np.shape(x)[-1]
y_dims = np.shape(y)[-1]
loss = 'MSPE'
hidden_dims = (40,40,40)
num_nets = 7
num_elites = 5
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
epochs = [15, 100, 250, 1000, 5000]
fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace':0.1, 'wspace':0.1})
ax_iter = axes.ravel().tolist()
data_ax = ax_iter[0]
kl_ax = ax_iter[1]


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
                        decay=1e-7,
                        max_logvar=.5,
                        min_logvar=-10,
                        session=sess)

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run([init_g, init_l])

test_bs = 1000
test_x = np.linspace(-15,15, num=test_bs)


for epoch in epochs:
    model_metrics = model.train(x, 
                            y,
                            batch_size=128, #512
                            max_epochs=epoch+1, # max_epochs 
                            min_epoch_before_break=epoch, # min_epochs, 
                            holdout_ratio=0.2, 
                                )

    pred_mean, pred_var = model.predict(test_x[...,None], factored=True, inc_var=True)
    pred_stds = np.sqrt(pred_var)
    pred_min, pred_max = pred_mean-pred_stds, pred_mean+pred_stds

    el_inds = model.elite_inds #[model.elite_inds[0], model.elite_inds[-3], model.elite_inds[-1]]        ## only returns elite indices
    elite_means, elite_mins, elite_maxs, elite_stds = pred_mean[el_inds], \
                                            pred_min[el_inds], \
                                            pred_max[el_inds], \
                                            pred_stds[el_inds]

    ###### measure model disagreement #######

    dkls = average_dkl(elite_means, elite_stds)
    dkls = np.clip(dkls, 0, 2)
    mean_vars = np.var(elite_means, axis=0)
    mean_vars *= np.var(mean_vars, axis=0)[None]
    print('done')

    kl_ax.plot(test_x, dkls, label=f"Avg. ens. KL Divergence@{epoch}")
    # kl_ax.plot(test_x, mean_vars, label=f"norm. var. of ens. means@{epoch}")

for i, elite_mean, elite_min, elite_max in zip(range(len(elite_means)), elite_means, elite_mins, elite_maxs):
    elite_mean, elite_min, elite_max = np.squeeze(elite_mean), np.squeeze(elite_min), np.squeeze(elite_max)
    data_ax.plot(test_x, elite_mean, color=COLOR[i], label=f'Model {i} w. Std.')  # Plot some data on the axes.
    data_ax.fill_between(test_x, elite_min, elite_max, alpha=0.2, color=COLOR[i])

kl_ax.scatter(x, y/np.std(y), s=0.1, color='green', label="Training (X,Y)")
kl_ax.legend()

data_ax.scatter(x, y, s=0.5, color='green', label="Training (X,Y)")
data_ax.legend()

# fig.legend()
# ax.plot(test_x, test_y)  # Plot some data on the axes.
# plt.scatter(test_x, pred_mean)

plt.show()
