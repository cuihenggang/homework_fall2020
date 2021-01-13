import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            # Add one more batch dimension
            observation = obs[None]

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        qa_values = self.critic.qa_values(observation)
        assert len(qa_values.shape) == 2
        action = np.argmax(qa_values, axis=1)

        # Remove the batch dimension
        return action.squeeze()
