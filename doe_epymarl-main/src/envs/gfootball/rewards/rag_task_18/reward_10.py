import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to enhance the effectiveness of central midfield play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints for positive transitions in gameplay
        self._transition_rewards = 0.05  # Reward for improved ball possession
        self._pace_rewards = 0.03        # Reward for maintaining optimal pacing

    def reset(self):
        """Resets the environment and sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for serialization."""
        to_pickle['checkpoint_reward_wrapper'] = {}  # Additional state from reward wrapper if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialized object."""
        from_pickle = self.env.set_state(state)
        # Load any required state configurations for wrapper here
        return from_pickle

    def reward(self, reward):
        """Modifies the reward depending on midfield performance."""
        components = {"base_score_reward": reward.copy(), "transition_reward": [], "pace_reward": []}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for agent_idx, o in enumerate(observation):
            trans_rew = 0
            pace_rew = 0
            
            # Assuming central midfielders are indexed 4 and 5 in a typical 11 player format
            midfielders = [4, 5]  
            
            if o['active'] in midfielders:
                # Reward for maintaining optimal pacing and ball control transitions
                if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                    # Ball control in terms of pacing and fluidity
                    pass_pace_eval = np.linalg.norm(o['ball_direction'])
                    if pass_pace_eval < 0.01:  # Assuming a small threshold indicating controlled pace
                        pace_rew = self._pace_rewards
                    # Improving transitions
                    if o['game_mode'] in [1,  3, 4]:  # Game modes like kick-off, free-kick, or corner
                        trans_rew = self._transition_rewards
                        
            components['transition_reward'].append(trans_rew)
            components['pace_reward'].append(pace_rew)
            
            # Summing up rewards for each agent
            reward[agent_idx] += trans_rew + pace_rew

        return reward, components

    def step(self, action):
        """Takes a step in the environment, adjusts rewards, and returns observations and adjusted rewards."""
        # Take an environment step
        obs, reward, done, info = self.env.step(action)
        # Modify the reward using the custom reward function
        reward, components = self.reward(reward)
        # Calculate and report the composite reward
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return obs, reward, done, info
