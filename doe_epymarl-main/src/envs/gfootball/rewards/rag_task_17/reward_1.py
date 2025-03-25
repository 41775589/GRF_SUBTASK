import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper adding rewards for mastering Wide Midfield responsibilities.
    This focuses on high passes and off-ball movement, emphasizing actions that 
    stretch the opposition's defense and create open spaces.
    """
    def __init__(self, env):
        super().__init__(env)
        self.high_pass_counter = np.zeros(10, dtype=int)  # Track high passes per episode
        self.position_reinforcements = np.zeros(10, dtype=int)  # Tracking off-ball movements
        self.high_pass_reward = 0.3
        self.position_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions counter

    def reset(self):
        """
        Reset the environment and counters.
        """
        self.high_pass_counter = np.zeros(10, dtype=int)  
        self.position_reinforcements = np.zeros(10, dtype=int)  
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state when pickling.
        """
        to_pickle['high_pass_counter'] = self.high_pass_counter
        to_pickle['position_reinforcements'] = self.position_reinforcements
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state when unpickling.
        """
        from_pickle = self.env.set_state(state)
        self.high_pass_counter = from_pickle['high_pass_counter']
        self.position_reinforcements = from_pickle['position_reinforcements']
        return from_pickle

    def reward(self, reward):
        """
        Customize the reward function considering high passes and strategic positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": np.zeros(len(reward)),
            "position_reward": np.zeros(len(reward))
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for executing high passes
            if 'ball_owned_team' in o and o['ball_owned_team'] in [0, 1]:
                # Assuming action index for high_pass is 5
                if o['sticky_actions'][5] == 1:
                    reward[rew_index] += self.high_pass_reward
                    components["high_pass_reward"][rew_index] += self.high_pass_reward

            # Reward for strategic off-ball positioning to stretch defense
            player_x, player_y = o['right_team'][rew_index]
            if player_x > 0.5 and abs(player_y) > 0.3:  # Wide and forward positions
                reward[rew_index] += self.position_reward
                components["position_reward"][rew_index] += self.position_reward

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
