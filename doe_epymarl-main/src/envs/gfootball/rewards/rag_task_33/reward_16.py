import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a long-range shooting reward, encouraging shots from outside the penalty box while penalizing lost ball possession in opponent's half."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_shot_reward": [0.0] * len(reward),
                      "possession_loss_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
                # Encourage shooting from distance: outside the penalty box
                if np.abs(o['ball'][0]) > 0.7:  # penalty box roughly at x > 0.7 or x < -0.7
                    components["long_shot_reward"][rew_index] += 0.5

                # Penalize losing possession in the opponent's half
                possession_lost = o['ball_owned_team'] == -1 and np.abs(o['ball'][0]) > 0
                if possession_lost:
                    components["possession_loss_penalty"][rew_index] -= 0.5

            # Update the reward for current agent with additional components
            reward[rew_index] = (reward[rew_index] +
                                 components["long_shot_reward"][rew_index] +
                                 components["possession_loss_penalty"][rew_index])

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
