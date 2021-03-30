from fake_env import FakeEnv
import gym

from utils import average_dkl, median_dkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

steps_per_epoch = 1000
epochs = 20
render = True

dyn_loss = 'NLL'
hidden_dims = (200,200,200)
bs = 1024

H = 25

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

real_env = gym.make('HalfCheetah-v2')
## create fake environment for model
fake_env = FakeEnv(real_env,
                    num_networks=7, 
                    num_elites=3, 
                    hidden_dims=hidden_dims,
                    learn_cost=False,
                    session = sess)

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run([init_g, init_l])

#############################
### sample data and train ###
#############################
obs_buf = []
nobs_buf = []
a_buf = []
r_buf = []
t_buf = []

for ep in range(epochs):
    o = real_env.reset()
    for st in range(steps_per_epoch):
        a = np.random.normal(loc=0, scale=1, size=real_env.action_space.shape)
        no, r, t, i = real_env.step(a)

        if render:
            real_env.render()

        obs_buf.append(o)
        nobs_buf.append(no)
        a_buf.append(a)
        r_buf.append(r)
        t_buf.append(t)

        o = no
        if t:
            real_env.reset()

    samples = dict(
        observations= np.stack(obs_buf),
        next_observations= np.stack(nobs_buf),
        actions= np.stack(a_buf),
        rewards= np.stack(r_buf),
        terminals= np.stack(t_buf)
    )

    fake_env.train_dyn_model(
        samples,
        batch_size=bs,
        holdout_ratio=0.2,
    )

    ##################################
    ### generate fake trajectories ###
    ##################################
    f_o = obs_buf[np.random.randint(len(obs_buf)-1)]
    for h in range(H):
        f_a = np.random.normal(loc=0, scale=1, size=real_env.action_space.shape)
        f_no,f_rew,f_term,f_info = fake_env.step(f_o, f_a, deterministic=True)
        
        ens_dkl_mean = f_info['ensemble_dkl_mean']
        f_o = f_no


