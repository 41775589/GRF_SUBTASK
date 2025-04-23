import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successfully executing high passes from midfield
    aimed at creating direct scoring opportunities."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_threshold = 0.2  # X coordinate threshold to define midfield area.
        self.high_pass_bonus = 0.5  # Reward addition for successful high pass from midfield.
        self.successful_passes = 0  # Counter for successful passes.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['successful_passes'] = self.successful_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.successful_passes = from_pickle.get('successful_passes', 0)
        return from_pickle

    def reward(self, reward):
        # Fetch the current observation from the environment.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        # Loop through the observations for each player
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate the conditions for a midfield high pass attempt.
            if (abs(o['ball'][0]) < self.midfield_threshold and  # Ball in the midfield
                o['ball_owned_team'] == o['designed'] and  # Ball owned by this team
                o['ball_direction'][2] > 0.1):  # Ball moving upwards (indicative of a high pass)

                # Rewarding the pass if it leads directly to a shot attempt near the goal.
                if (abs(o['ball'][0]) > 0.7):  # Ball ends up near the opponent's goal
                    components['high_pass_reward'][rew_index] = self.high_pass_bonus
                    reward[rew_index] += components['high_pass_reward'][rew_index]
                    self.successful_passes += 1

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
                self.sticky_actions_counter[i] += int(action)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
