import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tactical reward for mastering sliding tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {}
        return state

    def set_state(self, state):
        checkpoint_state = state['CheckpointRewardWrapper']
        # Restore the state needed
        return self.env.set_state(state)

    def reward(self, reward):
        # Access the environmental observations
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observations is None:
            return reward, components

        for i, observation in enumerate(observations):
            components.setdefault("tactical_reward", [0.0, 0.0])
            
            # Check if the ball is in the defensive third and a sliding action is executed
            ball_position = observation['ball'][0]  # Horizontal position of the ball
            sliding_action = observation['sticky_actions'][9]  # Assuming index 9 represents the sliding action

            # Define the defensive third for sliding tackles
            is_defensive_third = ball_position < -0.33

            # Reward an effective sliding tackle in the defensive third
            if is_defensive_third and sliding_action:
                components["tactical_reward"][i] += 1.0

            # Aggregate final rewards
            reward[i] += components["tactical_reward"][i]

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
