import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on quick counter-attack strategies 
    by promoting fast ball recovery and movement toward the opponent's goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Resets the environment and the sticky actions counter. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Stores the wrapped environment's state. """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restores the wrapped environment's state. """
        return self.env.set_state(state)

    def reward(self, reward):
        """ Reward modification to incentivize fast recovery and counter-attack.
        
        This function increases reward based on:
        1. Recovering ball possession near own goal and quickly transitioning it towards the midfield.
        2. Further incentives are allocated for pushing the ball to the opponent's half after recovery.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "recovery_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        for idx, obs in enumerate(observation):
            ball_position_x = obs['ball'][0]

            if obs['ball_owned_team'] == 0:  # If the left team has the ball
                if ball_position_x < -0.5:  # Ball is in team's own half
                    components['recovery_reward'][idx] = 0.2
                    # Further reward for progressing the ball to the opponent's half
                    if ball_position_x > 0:
                        components['counter_attack_reward'][idx] = 1.0

            reward[idx] += components['recovery_reward'][idx] + components['counter_attack_reward'][idx]

        return reward, components

    def step(self, action):
        """Step function to output modified reward and detailed info announcement."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                if action_flag:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
