import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for quick decision-making and efficiency in ball handling for counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_possession = -1

    def reset(self):
        """Reset the environment and the internal sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_possession = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'previous_possession': self.previous_possession}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_possession = from_pickle['CheckpointRewardWrapper']['previous_possession']
        return from_pickle

    def reward(self, reward):
        """Modify the reward to reflect efficient counter-attacking and decision-making."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward quick regain of ball possession for initiating counter-attacks
            if o['ball_owned_team'] == 0:  # Assuming our agent's team is 0
                if self.previous_possession != 0:
                    # Gained possession this frame
                    components['counter_attack_reward'][rew_index] = 1.0
                self.previous_possession = 0
            else:
                self.previous_possession = o['ball_owned_team']

            # Rewarded based on quickly transitioning to attack mode
            if components['counter_attack_reward'][rew_index] > 0:
                reward[rew_index] += components['counter_attack_reward'][rew_index]

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
