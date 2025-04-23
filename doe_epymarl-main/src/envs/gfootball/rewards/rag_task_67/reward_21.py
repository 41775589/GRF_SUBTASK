import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on the transition from defense to attack,
    encouraging skills like Short Pass, Long Pass, and Dribble.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Resets the environment and the sticky actions counter. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Adds additional rewards for successful short passes, long passes, and dribbles
        when moving from defense to attack.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check the ball possession and the game mode to promote the attack.
            if o['ball_owned_team'] == 1:  # Assuming '1' is the team id for the agent team
                current_action = o['sticky_actions']

                # Encourage passing and dribbling in the attacking half
                if o['ball'][0] > 0:  # Ball is on the opponent's half
                    if current_action[0] == 1 or current_action[4] == 1:  # Short/Long pass
                        components["pass_reward"][rew_index] = 0.1
                    if current_action[9] == 1:  # Dribbling
                        components["dribble_reward"][rew_index] = 0.1

            # Update the final reward for this step
            reward[rew_index] += components["pass_reward"][rew_index] + components["dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        """ Steps the environment and collects rewards and observations. """
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
