import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on strategic defensive positioning and quick transition capabilities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to control the defensive and transition rewards
        self._defensive_position_reward = 0.02
        self._quick_transition_reward = 0.05
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Compute the customized reward based on player positioning and transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "quick_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Encourage defensive positioning near own goal
            if obs['ball_owned_team'] == 1:  # If opponent controls the ball
                if obs['right_team_roles'][obs['active']] < 5:  # If player is defender
                    distance_to_goal = abs(obs['ball'][0] - (-1))  # X-axis distance to own goal
                    if distance_to_goal < 0.2:
                        components["defensive_positioning_reward"][i] = self._defensive_position_reward
                        reward[i] += components["defensive_positioning_reward"][i]

            # Encourage quick transition from defense to attack
            if obs['ball_owned_team'] == 0:  # If agent's team controls the ball
                ball_speed = np.linalg.norm(obs['ball_direction'][:2])  # Speed in xy-plane
                if ball_speed > 0.01:  # Considered 'fast' movement
                    components["quick_transition_reward"][i] = self._quick_transition_reward
                    reward[i] += components["quick_transition_reward"][i]

        return reward, components

    def step(self, action):
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
