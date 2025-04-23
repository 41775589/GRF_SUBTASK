import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards for practicing high passes with precision."""
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define high pass action
        self.high_pass_action = 6  # Assuming index 6 corresponds to the action for high passes in the action set

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            ball_ownership = (player_obs['ball_owned_team'] == player_obs['active']) and (player_obs['ball_owned_player'] >= 0)
            
            # Check if a high pass is executed
            performing_high_pass = ball_ownership and (self.sticky_actions_counter[self.high_pass_action] > 0)

            if performing_high_pass:
                # We reward high pass based on accuracy and timing
                ball_position = player_obs['ball']
                ball_destination = player_obs['ball_direction']

                # Projected impact point ideally should be near one of the opponent or in open space
                # This is simplified for example purposes
                ideal_zones = [(0.9, 0.0), (0.9, 0.42), (1.0, -0.42)]  # Likely zones near the opponent's goal
                accuracy_dist = min(np.linalg.norm(np.subtract(ideal_zones, ball_position + ball_destination)), key=np.linalg.norm)

                # Compute reward: Encourage accurate high passes
                high_pass_reward = (0.5 - accuracy_dist) * 2.0  # reward scale adjustment
                high_pass_reward = max(high_pass_reward, 0)  # reward should not be negative

                components['high_pass_reward'][rew_index] = high_pass_reward
                
                # Update the reward based on execution
                reward[rew_index] += components['high_pass_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Track sticky actions (e.g., high pass might be a maintained action)
        self.sticky_actions_counter = np.sum([obs[i]['sticky_actions'] for i in range(len(obs))], axis=0)
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
