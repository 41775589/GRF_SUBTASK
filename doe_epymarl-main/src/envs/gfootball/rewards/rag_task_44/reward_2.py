import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successfully executing the Stop-Dribble under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.zeros(2)
        self.stop_dribble_reward = 1.0
        self.ball_control_reward = 0.5
        self.stress_threshold = 0.3  # Example threshold distance to consider pressure
        self.ball_owned_team_last_step = -1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.zeros(2)
        self.ball_owned_team_last_step = -1
        return self.env.reset()

    def reward(self, reward):
        base_reward = reward
        stop_dribble_reward = [0.0] * len(reward)
        ball_control_reward = [0.0] * len(reward)

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': base_reward,
                            'stop_dribble_reward': stop_dribble_reward,
                            'ball_control_reward': ball_control_reward}

        for i in range(len(reward)):
            o = observation[i]
            current_player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]

            # Check proximity of opponents to determine pressure
            opponent_team = 'right_team' if o['ball_owned_team'] == 0 else 'left_team'
            proximity = min([np.linalg.norm(player - current_player_pos) for player in o[opponent_team]])

            # Reward stopping dribble under pressure
            if proximity < self.stress_threshold and 'sticky_actions' in o:
                if self.ball_owned_team_last_step == o['ball_owned_team'] and not o['sticky_actions'][8]:  # Assuming index 8 is dribble
                    stop_dribble_reward[i] = self.stop_dribble_reward

            # Reward for maintaining control of the ball
            if np.array_equal(self.previous_ball_position, o['ball']):
                ball_control_reward[i] = self.ball_control_reward

            reward[i] += stop_dribble_reward[i] + ball_control_reward[i]

        self.previous_ball_position = o['ball']
        self.ball_owned_team_last_step = o['ball_owned_team']

        reward_components = {
            'base_score_reward': base_reward,
            'stop_dribble_reward': stop_dribble_reward,
            'ball_control_reward': ball_control_reward
        }
        return reward, reward_components

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
