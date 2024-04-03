import ray
from ray.rllib.algorithms.ppo import PPO
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from gym import spaces
import or_gym
from or_gym.utils import create_env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tf_api, tf_original, tf_version = try_import_tf(error = True)
TUNE_DISABLE_AUTO_CALLBACK_SYNCER = 1

class KP0ActionMaskModel(TFModelV2):
     
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, true_obs_shape=(11,),
        action_embed_size=5, *args, **kwargs):
        
        # true_obs_shape is going to match the size of the state. 
        # If we stick with our reduced KP, that will be a vector with 11 entries. 
        # The other value we need to provide is the action_embed_size, which is going to be the size of our action space (5)
         
        super(KP0ActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, 
            *args, **kwargs)
         
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), 
                action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
 
    def forward(self, input_dict, state, seq_lens):
        
        # The actual masking takes place in the forward method where we unpack the mask, actions, and state from 
        # the observation dictionary provided by our environment. The state yields our action embeddings which gets 
        # combined with our mask to provide logits with the smallest value we can provide. 
        # This will get passed to a softmax output which will reduce the probability of selecting these actions to 0, 
        # effectively blocking the agent from ever taking these illegal actions.
        
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})
        intent_vector = tf_api.expand_dims(action_embedding, 1)
        action_logits = tf_api.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf_api.maximum(tf_api.log(action_mask), tf_api.float32.min)
        return action_logits + inf_mask, state
 
    def value_function(self):
        return self.action_embed_model.value_function()

  
# Configuration for gym environment
env_config = {'N': 5,
              'max_weight': 15,
              'item_weights': np.array([1, 12, 2, 1, 4]),
              'item_values': np.array([2, 4, 2, 1, 10]),
              'mask': True}
 
env_name = 'Knapsack-v0'
env = or_gym.make('Knapsack-v0', env_config=env_config)
 
print("Max weight capacity:\t{}kg".format(env.max_weight))
print("Number of items:\t{}".format(env.N))

# Register the model for Rllib usage
ModelCatalog.register_custom_model('kp_mask', KP0ActionMaskModel)
# Register the environment, so that we have a Trainable instance later
# ATTENTION: Tune needs the base class, not an instance of the environment like we get from or_gym.make(env_name) to work with. So we need to pass this to register_env using a lambda function as shown below.
env = create_env(env_name)
tune.register_env(env_name, lambda env_name: env(env_name, env_config=env_config))

trainer_config = {
    "model": {
        "custom_model": "kp_mask"   # Here we must use the custom model name taken in register process before
        },
    "env_config": env_config,       # env config from (or_)gym
    #"framework" : "tfe"             # tip by rllib to enable TensorFlow eager exection
     }

# ray.shutdown() maybe necessary in case of blocking instance
ray.init( ignore_reinit_error = True )
trainer = algorithms.ppo.PPO(env='Knapsack-v0', config=trainer_config)

env = trainer.env_creator('Knapsack-v0')
state = env.state

# Use the action masking to disable the agent to take specific actions, i.e. to avoid taking element in knapsack by index
# state['action_mask'][0] = 0

# Train an agent for 1000 states 
actions = np.array([trainer.compute_single_action(state) for i in range(10000)])

# If action masking used, check that this action was never taken
# print(any(actions==0))
