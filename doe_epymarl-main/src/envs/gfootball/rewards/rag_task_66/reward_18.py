import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective short passes under pressure, 
    focusing on ball retention and effective distribution for defensive stability 
    and counter-attack transitions."""

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
        """
        Enhance the reward signal based on successful short passes under defensive pressure, 
        particularly focusing on maintaining possession as a metric of success.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "short_pass_reward": [0.0] * len(reward)}

        for rew_index, (agent_reward, o) in enumerate(zip(reward, observation)):
            # If the game mode is not normal (e.g., set piece), reduce the incentives for passes
            if o['game_mode'] != 0:
                continue

            # Bonus for successful short pass while being close to opponents
            if o['ball_owned_team'] == 0:  # Assuming our agent is on the left team
                distance_to_nearest_opponent = np.min(np.linalg.norm(o['left_team'][o['ball_owned_player']] - o['right_team'], axis=1))
                if distance_to_nearest_opponent < 0.2:  # Arbitrary threshold for "pressure"
                    # Check if a short pass is effectively used
                    if o['sticky_actions'][6] == 1 or o['sticky_actions'][7] == 1:  # action pass is active
                        components["short_pass_reward"][rew_index] = 0.2
                        agent_reward += components["short_pass_reward"][rew_index]

        reward, components = self.reward(reward)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()  # re-obtaining the updated observations
        return observation, reward, done, info
