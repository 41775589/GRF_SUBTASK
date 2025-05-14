import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on positional control and passing for mastery in integrating midfield and defensive play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defining sectors on the field for reward transitions, -1 to 0 for left side, 0 to 1 for right.
        self.passing_thresholds = np.linspace(-1, 1, num=3)  # Sectioning the field into three zones
        self.passing_bonus = 0.1  # Bonus reward for successful passes
        self.position_control_bonus = 0.05  # Bonus for ball control in midfield

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward}

        components = {
            "base_score_reward": reward.copy(),
            "positional_control_reward": [0.0],
            "passing_reward": [0.0]
        }

        o = observation[0]
        ball_position = o['ball'][0]  # x-coordinate of the ball
        ball_owned = o['ball_owned_team'] == 0  # team 0 is our team
        active_player_pos = o['left_team'][o['active']][0]  # x-coordinate of active player

        # Reward controlling the ball in defined midfield area
        if -0.3 <= ball_position <= 0.3:
            components["positional_control_reward"][0] += self.position_control_bonus
            reward[0] += components["positional_control_reward"][0]

        # Reward for passes within thresholds for team 0
        if ball_owned:
            last_position = self.passing_thresholds[np.digitize(active_player_pos, self.passing_thresholds) - 1]
            ball_travel = ball_position - last_position
            if ball_travel > 0.1:  # assuming ball successfully moved forward to another player
                components["passing_reward"][0] += self.passing_bonus
                reward[0] += components["passing_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()  # Ensuring observation is updated
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active

        return observation, reward, done, info
