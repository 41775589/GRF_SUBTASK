import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic midfield control and precise defense management."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_control_scale = 0.1
        self.defensive_action_scale = 0.15
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """Resets internal states and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        """Save state of the wrapper along with environment state."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state of the wrapper along with environment state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Calculates adjusted reward based on midfield control and defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_control_reward": [0.0] * len(reward),
            "defensive_action_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0:  # If own team has the ball
                if -0.2 <= obs['ball'][0] <= 0.2:  # Ball in the midfield area
                    components["midfield_control_reward"][i] = self.midfield_control_scale
                    reward[i] += components["midfield_control_reward"][i]

            if obs['game_mode'] in (2, 3, 4, 6):  # Defensive game modes (goal-kick, free-kick, corner, penalty)
                components["defensive_action_reward"][i] = self.defensive_action_scale
                reward[i] += components["defensive_action_reward"][i]
        
        return reward, components

    def step(self, action):
        """Environment step with custom reward calculation."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for i, agent_obs in enumerate(obs):
            self.sticky_actions_counter += agent_obs['sticky_actions']
            for j, stick_action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{j}"] = stick_action
        return observation, reward, done, info
