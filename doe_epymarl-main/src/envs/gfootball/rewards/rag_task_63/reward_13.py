import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specific reward function for training goalkeepers.
    This reward function focuses on shot stopping, decision-making during ball
    distribution, and communication with defenders.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the current state of the environment including wrapper-specific data.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment and retrieve wrapper-specific data.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward function to focus on goalkeeper training aspects.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for idx, rew in enumerate(reward):
            if observation is None:
                continue

            o = observation[idx]
            goalkeeper_saves = (o['game_mode'] == 6 and o['ball_owned_team'] == 1 and
                                o['right_team_roles'][o['active']] == 0)  # 0 is typically goalkeeper role
            
            effective_distribution = (
                o['ball_owned_team'] == 1 and
                o['ball_owned_player'] == o['active'] and
                np.linalg.norm(o['ball_direction'][:2]) > 0.5)  # Simulate effective distribution by distance

            communication_bonus = (o['ball_owned_team'] == 1 and
                                   np.any(o['right_team_yellow_card']))  # Keeping control despite pressure

            # Calculate additional rewards
            components['goalkeeper_saves'] = 1.0 if goalkeeper_saves else 0.0
            components['effective_distribution'] = 0.5 if effective_distribution else 0.0
            components['communication_bonus'] = 0.2 if communication_bonus else 0.0

            # Summing rewards
            total_additional_reward = (components['goalkeeper_saves'] +
                                       components['effective_distribution'] +
                                       components['communication_bonus'])

            reward[idx] += total_additional_reward
        return reward, components

    def step(self, action):
        """
        Execute a step in the environment, update the reward using the
        modified reward function, and provide additional information.
        """
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
