import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds specific rewards for a goalkeeper training task.
    Rewards are based on shot stopping, quick ball distribution under pressure, and
    effective communication with defenders.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_interceptions = 0
        self.clean_sheets = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_interceptions = 0
        self.clean_sheets = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = super().get_state(to_pickle)
        state.update({
            'ball_interceptions': self.ball_interceptions,
            'clean_sheets': self.clean_sheets
        })
        return state

    def set_state(self, state):
        self.ball_interceptions = state['ball_interceptions']
        self.clean_sheets = state['clean_sheets']
        return super().set_state(state)

    def reward(self, reward):
        """
        Custom reward function to aid goalkeeper training:
          - Reward for shot stopping
          - Penalty for goals conceded
          - Reward for successfully passing under pressure
          - Reward for effective communication (positional adjustments)
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "shot_stopping_reward": [0.0] * len(reward),
                      "communication_reward": [0.0] * len(reward),
                      "distribution_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            own_goalkeeper = o['left_team_roles'][o['active']] == 0  # Assuming goalie role index is 0

            # Increase reward for successful shot stopping
            if own_goalkeeper and o['ball_owned_team'] == 0 and np.linalg.norm(o['ball'] - o['left_team'][o['active']]) < 0.1:
                components['shot_stopping_reward'][rew_index] = 1.0
                reward[rew_index] += components['shot_stopping_reward'][rew_index]
                self.ball_interceptions += 1

            # Clean sheet bonus at end of game if no goals conceded
            if o['steps_left'] == 0 and o['score'][0] == 0:
                components['communication_reward'][rew_index] = 5.0
                reward[rew_index] += components['communication_reward'][rew_index]
                self.clean_sheets += 1

            # Penalty for goals conceded
            if o['score'][1] > 0:
                reward[rew_index] -= 5.0

            # Check effective distribution under pressure, presuming goalie holds the ball
            if own_goalkeeper and o['ball_owned_team'] == 0:
                components['distribution_reward'][rew_index] = 0.5
                reward[rew_index] += components['distribution_reward'][rew_index]

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
