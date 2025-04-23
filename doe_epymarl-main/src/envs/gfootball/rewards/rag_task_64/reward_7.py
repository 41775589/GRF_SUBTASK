import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing high crosses and passes effectively.
    This promotes high passes, crossing from varying distances and angles.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the wrapper state with the environment's reset.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Enhance the reward based on high passes (action 9) and the position on the field.
        More reward for crosses closer to opponent's goal and from tricky angles.
        """
        observation = self.env.unwrapped.observation()
        base_score_reward = np.copy(reward)
        high_pass_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            # Encourage high passes when enabled
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Assuming index 9 is high pass
                player_pos = o['right_team'][o['active']] if o['active'] >= 0 else [0, 0]
                # Closer to the opponent's goals on the x-axis
                proximity_to_goal = np.abs(player_pos[0] - 1)  
                angle_factor = np.abs(player_pos[1])  # Challenging angles have higher rewards

                high_pass_reward[idx] = 0.3 * (1 - proximity_to_goal) + 0.1 * angle_factor
                reward[idx] += high_pass_reward[idx]

        components = {"base_score_reward": base_score_reward, "high_pass_reward": high_pass_reward}
        return reward, components

    def get_state(self, to_pickle):
        """
        Save additional state information for this wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore state from saved information, specific to this wrapper.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def step(self, action):
        """
        Steps through environment with action, applying the reward wrapper.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        
        # To simulate sticky actions counting which is left out in the original prompt but evidentially important
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
