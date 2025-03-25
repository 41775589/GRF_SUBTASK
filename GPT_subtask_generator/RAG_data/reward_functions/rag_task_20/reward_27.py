import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds an offensive strategy reward focusing on team coordination and reaction."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions
        self.zone_rewards_collected = {}  # To track rewards collected for advancing to zones

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['zone_rewards_collected'] = self.zone_rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.zone_rewards_collected = from_pickle.get('zone_rewards_collected', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "zone_advancement_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components["zone_advancement_reward"][rew_index] = self.calculate_zone_reward(o, rew_index)

            # Summing the base score and additional components to shape the final reward for each agent
            reward[rew_index] += components["zone_advancement_reward"][rew_index]

        return reward, components

    def calculate_zone_reward(self, observation, index):
        """Calculate the reward given to agent based on zone advancement and ball control."""
        zone_reward = 0.0
        if observation['ball_owned_team'] == 1 and observation['ball_owned_player'] != -1:
            # Calculate rewards based on the position of the ball on the field scaled by progress towards goal
            ball_x = observation['ball'][0]
            if ball_x > 0.5:  # Past midfield towards opponent's goal
                zone = int(ball_x * 10) - 5
                if zone not in self.zone_rewards_collected.get(index, []):
                    zone_reward = 0.1 * (zone + 1)  # Reward scales with proximity to goal
                    if index not in self.zone_rewards_collected:
                        self.zone_rewards_collected[index] = []
                    self.zone_rewards_collected[index].append(zone)
        return zone_reward

    def step(self, action):
        # This method remains unchanged as advised.
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
