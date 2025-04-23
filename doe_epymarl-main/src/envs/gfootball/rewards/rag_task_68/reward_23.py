import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on offensive strategies including shooting,
    dribbling, and passing in efficient ways to score or assist in an open play scenario.
    """

    def __init__(self, env):
        super().__init__(env)
        self.shooting_bonus = 0.5
        self.dribbling_bonus = 0.2
        self.passing_bonus = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointReward'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointReward']
        return from_pickle

    def reward(self, reward):
        # Reward modification with components setup
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # Check if the right team owns the ball
                ball_owner = o['ball_owned_player']
                action = o['sticky_actions']

                # Shooting at goal
                if o['game_mode'] in [0, 3] and action[9] and o['ball'][0] > 0.8:  # agent intends to shoot in opponent's half
                    components['shooting_reward'][idx] = self.shooting_bonus * (0.8 + abs(o['ball'][1]))
                    reward[idx] += components['shooting_reward'][idx]

                # Dribbling near opponents
                if action[9] and any(np.linalg.norm(o['right_team'][ball_owner] - p) < 0.1 for p in o['left_team']):  # close to any opponent
                    components['dribbling_reward'][idx] = self.dribbling_bonus
                    reward[idx] += components['dribbling_reward'][idx]

                # Long or high pass logic
                if action[0] or action[4] or action[6]:  # agent is passing in some direction
                    direction = np.arctan2(o['ball_direction'][1], o['ball_direction'][0])
                    if direction > np.pi / 4 and direction < 3 * np.pi / 4 and np.linalg.norm(o['ball_direction']) > 0.5:
                        # Considering high or long balls as effective passes
                        components['passing_reward'][idx] = self.passing_bonus
                        reward[idx] += components['passing_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
