import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on midfielders' playmaking and strikers' finishing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment and include additional state from the wrapper."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment and restore additional state from the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on strategic positioning and finishing plays."""
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': np.array(reward),  # original reward from the environment
            'playmaking_reward': np.zeros(len(reward)),
            'finishing_reward': np.zeros(len(reward))
        }

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            # Encourage midfielders to make effective passes (playmaking)
            if 4 in obs['left_team_roles']:  # Assuming role 4 is a midfielder role
                if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == 4:
                    components['playmaking_reward'][idx] += 0.1  # small reward for having the ball

            # Reward strikers for scoring goals (finishing)
            if 9 in obs['left_team_roles']:  # Assuming role 9 is a striker role
                scorer = obs.get('score', [0, 0])
                if scorer[0] > scorer[1]:  # Left team scored
                    components['finishing_reward'][idx] += 0.3  # greater reward for scoring

        # Update the overall reward
        updated_rewards = components['base_score_reward'] + components['playmaking_reward'] + components['finishing_reward']
        return updated_rewards, components 
        
    def step(self, action):
        """Take a step in the environment and apply reward modifications."""
        obs, reward, done, info = self.env.step(action)
        modified_reward, reward_components = self.reward(reward)
        info['final_reward'] = sum(modified_reward)
        
        # Include detailed reward information in the info dictionary for easy tracking
        for key, value in reward_components.items():
            info[f'component_{key}'] = sum(value)

        # Track sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action_value

        return obs, modified_reward, done, info
