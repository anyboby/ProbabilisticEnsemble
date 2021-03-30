import numpy as np
import tensorflow as tf
import pdb

from pe_factory import build_PE, format_samples_for_dyn, format_samples_for_cost
from utils import average_dkl, median_dkl

from itertools import count
import warnings
import time

EPS = 1e-8

class FakeEnv:

    def __init__(self, true_environment,
                    num_networks=7, 
                    num_elites = 5, 
                    hidden_dims = (220, 220, 220),
                    loss_type = 'NLL',
                    learn_cost=False,
                    session = None):
        
        self.env = true_environment
        self.obs_dim = np.prod(self.observation_space.shape)
        self.act_dim = np.prod(self.action_space.shape)

        self._session = session
        self.learn_cost = learn_cost
        self.rew_dim = 1
        self.cost_classes = [0,1]

        self.num_networks = num_networks
        self.num_elites = num_elites
        
        #### create fake env from model
        input_dim_dyn = self.obs_dim + self.act_dim
        input_dim_c = 2 * self.obs_dim + self.act_dim
        output_dim_dyn = self.obs_dim + self.rew_dim
        self.dyn_loss = loss_type

        self._model = build_PE(in_dim=input_dim_dyn, 
                                        out_dim=output_dim_dyn,
                                        name='DynEns',
                                        loss=self.dyn_loss,
                                        hidden_dims=hidden_dims,
                                        lr=1e-3,
                                        num_networks=num_networks, 
                                        num_elites=num_elites,
                                        use_scaler_in = True,
                                        use_scaler_out = True,
                                        decay=1e-6,
                                        max_logvar=.5,
                                        min_logvar=-10,
                                        session=self._session)
        if self.learn_cost:
            self.cost_m_loss = 'MSE'
            output_activation = 'softmax' if self.cost_m_loss=='CE' else None

            self._cost_model = build_PE(in_dim=input_dim_c, 
                                        out_dim=2 if self.cost_m_loss=='CE' else 1,
                                        loss=self.cost_m_loss,
                                        name='CostEns',
                                        hidden_dims=(64,64),
                                        lr=1e-4,
                                        output_activation = output_activation,
                                        num_networks=num_networks,
                                        num_elites=num_elites,
                                        use_scaler_in = False,
                                        use_scaler_out = False,
                                        decay=1e-6,
                                        session=self._session)            
        else:
            self._cost_model = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds
    
    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        obs_depth = len(obs.shape)
        if obs_depth == 1:
            obs = obs[None]
            act = act[None]
            return_single=True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)

        if obs_depth==3:
            inputs, shuffle_indxs = self.forward_shuffle(inputs)

        ens_dyn_means, ens_dyn_vars = self._model.predict(inputs, factored=True, inc_var=True)       #### dyn_vars gives ep. vars for 
                                                                                                    ## deterministic ensembles and al. var for probabilistic

        if obs_depth==3:
            ens_dyn_means, ens_dyn_vars = self.inverse_shuffle(ens_dyn_means, shuffle_indxs), self.inverse_shuffle(ens_dyn_vars, shuffle_indxs)
        
        ens_dyn_means[:,:,:-self.rew_dim] += obs           #### models output state change rather than obs completely
        ens_dyn_stds = np.sqrt(ens_dyn_vars)
        ens_dkl_path = np.mean(average_dkl(ens_dyn_means, ens_dyn_stds), axis=-1)
        ens_dkl_mean = np.mean(ens_dkl_path)
        ens_ep_var = np.var(ens_dyn_means, axis=0)
        
        if not deterministic:
            ens_dyn_means += ens_dyn_stds

        #### choose one model from ensemble randomly
        if obs_depth==3:
            samples = ens_dyn_means
        else:
            _, batch_size, _ = ens_dyn_means.shape
            model_inds = self._model.random_inds(batch_size)        ## only returns elite indices
            batch_inds = np.arange(0, batch_size)
            samples = ens_dyn_means[model_inds, batch_inds]
        ##########################

        #### retrieve next_obs, r, cost, and terms for new state
        next_obs = samples[...,:-self.rew_dim]
        rews = samples[...,-self.rew_dim:]

        if self.learn_cost:
            inputs_cost = np.concatenate((obs, act, next_obs), axis=-1)
            costs = self._cost_model.predict(inputs_cost, factored=False, inc_var=False)

            if self.cost_m_loss=='CE':
                costs = np.argmax(costs, axis=-1)[...,None]
        else:
            costs = np.zeros_like(rews)
        terminals = np.zeros_like(rews, dtype=np.bool)
        ##########################


        if return_single:
            next_obs = next_obs[0]
            rews = rews[0]
            costs = costs[0]
            terminals = terminals[0]

        info = {
                'ensemble_dkl_mean' : ens_dkl_mean,
                'ensemble_dkl_path' : ens_dkl_path,
                'ensemble_mean_var' : ens_dyn_vars.mean(),
                'ensemble_ep_var' : ens_ep_var,
                'rew':rews,
                'cost':costs,
                }

        return next_obs, rews, terminals, info

    def train_dyn_model(self, samples, **kwargs):
        #### format samples to fit: inputs: concatenate(obs,act), outputs: concatenate(rew, delta_obs)        
        train_inputs_dyn, train_outputs_dyn = format_samples_for_dyn(samples,
                                                                    )
        model_metrics = self._model.train(train_inputs_dyn, 
                                            train_outputs_dyn, 
                                            **kwargs,
                                            )
        return model_metrics

    def train_cost_model(self, samples, **kwargs):        
        #### format samples to fit: inputs: concatenate(obs,act,nobs), outputs: (cost,)
        assert self._cost_model
    
        inputs, targets = format_samples_for_cost(samples, 
                                                    one_hot=self.cost_m_loss=='CE',
                                                    )
        #### Useful Debugger line: np.where(np.max(train_inputs_cost[np.where(train_outputs_cost[:,1]>0.8)][:,3:54], axis=1)<0.95)

        cost_model_metrics = self._cost_model.train(inputs,
                                    targets,
                                    **kwargs,
                                    )                                            
            
        return cost_model_metrics

    def random_inds(self, size):
        return self._model.random_inds(batch_size=size)

    def reset_model(self):
        self._model.reset()
        if self._cost_model:
            self._cost_model.reset()
        
    def filter_elite_inds(self, data, n_elites, apply_too = None):
        '''
        extracts the closest data to the median
        data 0-axis is ensemble axis
        data 1-axis is batch axis
        apply_too: a list of arrays with same dims as data that the same filtration is applied to. 
        '''
        ### swap for convenience
        data_sw = np.swapaxes(data, 0, 1)
        mse_median = np.mean((data_sw-np.median(data_sw, axis=1)[:,None,...])**2, axis=-1)
        sorted_inds = np.argsort(mse_median, axis=1)[:, :n_elites]
        replace_inds = sorted_inds[:, 0:self.num_networks-n_elites]
        batch_inds = np.arange(data_sw.shape[0])[...,None]

        res = np.concatenate((data_sw[batch_inds, sorted_inds], data_sw[batch_inds, replace_inds]), axis=1)
        res = np.swapaxes(res, 0,1)

        if apply_too is not None:
            sw_list = [np.swapaxes(arr, 0, 1) for arr in apply_too]
            res_list_too = [np.concatenate((sw_too[batch_inds, sorted_inds], sw_too[batch_inds, replace_inds]), axis=1) for sw_too in sw_list]
            res_list_too = [np.swapaxes(res_too, 0,1) for res_too in res_list_too]
            return res, res_list_too
        return res

    def forward_shuffle(self, ndarray):
        """
        shuffles ndarray forward along axis 0 with random elite indices, 
        Returns shuffled copy of ndarray and indices with which was shuffled
        """
        idxs = np.random.permutation(ndarray.shape[0])
        shuffled = ndarray[idxs]
        return shuffled, idxs

    def inverse_shuffle(self, ndarray, idxs):
        """
        inverses a shuffle of ndarray forward along axis 0, given the used indices. 
        Returns unshuffled copy of ndarray
        """
        unshuffled = ndarray[idxs]
        return unshuffled

    def close(self):
        pass
