import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages teamwork and defensive strategy by adding a reward for coordination."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.team_spirit_bonus = 0.01
        self.defensive_bonus = 0.05
        self.ball_control_bonus = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        # Reward components initialization
        components = {
            'base_score_reward': reward.copy(),
            'team_spirit': [0.0],
            'defensive_positioning': [0.0],
            'maintaining_ball_control': [0.0]
        }

        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        # Calculate additional rewards
        for agent_id, o in enumerate(observation):
            if o is None:
                continue

            designated = o['designated']
            active_player = o['active']

            # Reward for maintaining ball control within the team
            if o['ball_owned_team'] == 0:  # Assuming the left team is our team
                components['maintaining_ball_control'][agent_id] += self.ball_control_bonus

            # Reward for players being in a good defensive position
            for player_pos in o['left_team']:
                if player_pos[0] < -0.5:  # More defensive positioning on their own half
                    components['defensive_positioning'][agent_id] += self.defensive_bonus

            # Reward for teamwork and passing
            sticky_actions = o['sticky_actions']
            if sticky_actions[8] == 1 or sticky_actions[9] == 1:  # Sprint or dribble, indicating good effort
                components['team_spirit'][agent_id] += self.team_spirit_bonus

        # Calculate final reward
        final_rewards = [components['base_score_reward'][i] +
                         components['team_spirit'][i] +
                         components['defensive_positioning'][i] +
                         components['maintaining_ball_control'][i]
                         for i in range(len(reward))]

        return final_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)
