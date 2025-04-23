import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for successful standing tackles that achieve ball possession
    without incurring fouls (yellow or red cards). This is aligned with focusing the training
    on precision and control during tackles in normal and set-piece scenarios.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize any necessary variables or counters here
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and any variables that are specific to the reward function.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save additional wrapper-related state.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        # Additional state can be saved here if necessary
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore wrapper-related state.
        """
        from_pickle = self.env.set_state(state)
        # Additional state can be retrieved and set here if necessary
        return from_pickle

    def reward(self, reward):
        """
        Modify the rewards given by the environment based on additional criteria.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),  # store base score
            "tackle_success_bonus": [0.0] * len(reward)
        }
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check for a successful tackle and possession gain without penalties
            if o['sticky_actions'][8] == 1:  # Assuming index 8 is the tackle action
                if o['ball_owned_team'] == 0 and (not o['left_team_yellow_card'][o['active']] and not o['left_team_yellow_card'][o['active']]):
                    # Reward for successful tackle without receiving a card
                    components["tackle_success_bonus"][rew_index] = 0.5
                    reward[rew_index] += components["tackle_success_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info.update({
            'final_reward': sum(reward),
            **{'component_' + key: sum(value) for key, value in components.items()},
        })
        
        # Track sticky actions activation
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
