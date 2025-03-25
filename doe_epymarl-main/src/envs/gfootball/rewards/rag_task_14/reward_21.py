import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides sweeper-specific rewards for clearing the ball from defensive zones and last-man tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defense_positioning_reward": [0.0] * len(reward),
            "ball_clearance_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            components["defense_positioning_reward"][rew_index] = self.calculate_positioning_reward(o)
            components["ball_clearance_reward"][rew_index] = self.calculate_clearance_reward(o)

            reward[rew_index] += (
                components["defense_positioning_reward"][rew_index] + 
                components["ball_clearance_reward"][rew_index]
            )
        return reward, components

    def calculate_positioning_reward(self, o):
        """Rewards players for being in a good defensive position when needed."""
        # Reward players for positioning behind the ball and in their own half if opposition owns the ball
        if o['ball_owned_team'] == 1:  # ball owned by the opposing team
            if o['active'][0] < 0 and o['ball'][0] > o['active'][0]:  # active player is behind the ball horizontally
                return 0.1  # Positive reward for defensive positioning
        return 0.0

    def calculate_clearance_reward(self, o):
        """Rewards players for clearing the ball away from the danger zone."""
        # Reward for clearing the ball from the defensive third of the field
        if o['ball_owned_team'] == 0 and abs(o['active'][0]) > 0.666:  # in defensive third
            # Assuming a rough action that represents clearing like a long ball or similar
            if self.sticky_actions_counter[0] or self.sticky_actions_counter[4]:  # using simplistic action checks
                return 0.3  # Reward for clearing action
        return 0.0

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
