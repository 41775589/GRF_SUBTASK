import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tailored reward for offensive strategies that emphasize coordination, passing, and strategic positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passing_reward_multiplier = 0.5
        self.positioning_reward_multiplier = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset wrapped environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        """Set states of the environment including the count of sticky actions."""
        from_state = self.env.set_state(state)
        self.sticky_actions_counter = from_state['sticky_actions']
        return from_state

    def reward(self, reward):
        """Customize reward function focusing on passing, positioning, and strategic team play."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": np.zeros(len(reward)),
            "positioning_reward": np.zeros(len(reward))
        }

        for i in range(len(reward)):
            obs = observation[i]
            if obs['ball_owned_team'] == 1 or obs['ball_owned_team'] == 0:
                if obs['ball_owned_player'] == obs['active']:
                    # Reward for maintaining possession
                    components["passing_reward"][i] += self.passing_reward_multiplier

                    # Add positioning reward based on how close to scoring position the player is
                    x_position = obs['left_team'][obs['active']][0] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']][0]
                    components["positioning_reward"][i] += max(0, self.positioning_reward_multiplier * (x_position))

            reward[i] += components["passing_reward"][i] + components["positioning_reward"][i]

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
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
