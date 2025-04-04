import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on defensive skills including interceptions and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_counter = {}

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the state of the environment for checkpoints."""
        to_pickle['intercept_counter'] = self.intercept_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the serialized state back to the environment."""
        from_pickle = self.env.set_state(state)
        self.intercept_counter = from_pickle.get('intercept_counter', {})
        return from_pickle

    def reward(self, reward):
        """Compute the reward for the current step based on defensive positioning and interceptions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i, o in enumerate(observation):
            # Reward for intercepting the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if self.intercept_counter.get(i, 0) == 0:
                    components["intercept_reward"][i] = 1.0
                    self.intercept_counter[i] = 1
                    
            # Reward based on defensive positioning
            if 'left_team_direction' in o and o['ball_owned_team'] == 1:
                my_pos = o['left_team'][o['active']]
                ball_pos = o['ball']
                distance_to_ball = np.linalg.norm(my_pos - ball_pos[:2])
                if distance_to_ball < 0.2:
                    components["positioning_reward"][i] = 0.5  # Closer to ball in defensive situation

            # Update rewards
            reward[i] += components["intercept_reward"][i] + components["positioning_reward"][i]

        return reward, components

    def step(self, action):
        """Apply actions, compute rewards, and return observations."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding detailed rewards to info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Record sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
