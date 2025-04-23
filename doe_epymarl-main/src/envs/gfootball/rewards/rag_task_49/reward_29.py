import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function specialized for shooting from central field positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Retrieve the current observation from the environment
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_reward": [0.0, 0.0]  # Using list of 2 because generally, two players are controlled
        }

        for idx, o in enumerate(observation):
            # Check if the player is in the central field zone (approx. x between -0.25 to 0.25)
            player_pos = o['right_team'][o['active']][0] if o['ball_owned_team'] == 1 else o['left_team'][o['active']][0]
            if -0.25 <= player_pos <= 0.25:
                # Reinforce shooting from the central field positions
                components["shooting_accuracy_reward"][idx] = 0.2

            # Adding the components to the base reward
            reward[idx] += components["shooting_accuracy_reward"][idx]

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
