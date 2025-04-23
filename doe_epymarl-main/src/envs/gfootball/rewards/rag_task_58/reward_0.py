import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive coordination and ball distribution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_checkpoint = 0.5  # Reward when the ball is controlled effectively at key points
        self.defensive_actions_checkpoint = 0.3  # Reward for effective defensive actions

    def reset(self):
        """Reset the environment and the counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get state to pickle."""
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        """Restore state from pickle."""
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward logic to incorporate teamwork and defensive coordination."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_control_reward": [0.0] * len(reward),
                      "defensive_actions_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            if o is None:
                continue
            
            if o['ball_owned_team'] == 0:  # Team 0 is the controlled team
                # Ball control in defensive half, encouraging maintaining possession in critical areas
                if o['ball'][0] <= 0: 
                    components["ball_control_reward"][i] = self.ball_control_checkpoint

            # Encourage defensive play by position and ball interception
            if any((np.linalg.norm(teammate - o['ball'][:2]) < 0.1 for teammate in o['left_team'])):
                components["defensive_actions_reward"][i] = self.defensive_actions_checkpoint

            # Calculate total reward
            reward[i] += components["ball_control_reward"][i] + components["defensive_actions_reward"][i]

        return reward, components

    def step(self, action):
        """Apply actions, step the environment, and compute custom reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
