import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_successful = 0.2  # Reward for successful tackle
        self.sliding_tackles_successful = 0.3  # Reward for successful sliding tackle
        self.interceptions_successful = 0.1  # Reward for successful interception

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "slide_tackle_reward": [0.0] * len(reward),
                      "interception_reward": [0.0] * len(reward)}
        
        if observation is None or 'left_team_roles' not in observation:
            return reward, components

        for rew_index in range(len(reward)):
            active_player = observation[rew_index]['active']
            if active_player == -1:
                continue

            player_role = observation[rew_index]['left_team_roles'][active_player]
            player_has_ball = observation[rew_index]['ball_owned_player'] == active_player

            # Check for successful tackles and interceptions
            if player_role in [2, 3, 4, 5] and player_has_ball:
                components['tackle_reward'][rew_index] = self.tackles_successful
                reward[rew_index] += components['tackle_reward'][rew_index]

            # Check for successful sliding tackles
            if player_role in [2, 3] and player_has_ball:
                components['slide_tackle_reward'][rew_index] = self.sliding_tackles_successful
                reward[rew_index] += components['slide_tackle_reward'][rew_index]

            # Check for successful interceptions
            if player_role in [5] and player_has_ball:
                components['interception_reward'][rew_index] = self.interceptions_successful
                reward[rew_index] += components['interception_reward'][rew_index]

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
            for i, action_item in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_item
        return observation, reward, done, info
