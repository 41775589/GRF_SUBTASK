import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward based on offensive tactics and team coordination for football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficients for different reward components
        self.passing_reward_coef = 0.2
        self.positioning_coef = 0.1
        self.shooting_coef = 1.0
        self.progress_coef = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "progress_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o is None:
                continue

            components["passing_reward"][rew_index] = self.passing_reward(o) * self.passing_reward_coef
            components["positioning_reward"][rew_index] = self.positioning_reward(o) * self.positioning_coef
            components["shooting_reward"][rew_index] = self.shooting_reward(o) if o['score'][0] > o['score'][1] else 0
            components["progress_reward"][rew_index] = self.progress_reward(o) * self.progress_coef
            
            reward[rew_index] += sum(components[k][rew_index] for k in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def passing_reward(self, o):
        """Calculate the reward for successful passes based on difference in possession."""
        pass_effectiveness = np.sum(o['sticky_actions'][8:10])  # Assuming actions for pass are in these slots
        return pass_effectiveness

    def positioning_reward(self, o):
        """Reward players for being well positioned relative to the ball and the goal."""
        player_pos = o['left_team'] if o['left_team_active'] else o['right_team']
        goal_pos = [1, 0] if o['left_team_active'] else [-1, 0]
        ball_pos = o['ball'][:2]

        dist_to_ball = np.linalg.norm(player_pos - ball_pos)
        dist_to_goal = np.linalg.norm(player_pos - goal_pos)

        return (1 / (dist_to_ball + 0.1)) + (1 / (dist_to_goal + 0.1))

    def shooting_reward(self, o):
        """Reward for shots on goal which result in a score increment."""
        if o['game_mode'] in {6} and o['ball_owned_team'] == o['active']:
            return self.shooting_coef
        return 0

    def progress_reward(self, o):
        """Generate a reward for making progress towards the opponent's goal."""
        if o['active'] == -1:
            return 0

        ball_start_pos = o['ball'][0]
        ball_end_pos = o['ball'][0] + o['ball_direction'][0]
        progress = ball_end_pos - ball_start_pos
        progress_reward = progress if o['left_team_active'] and progress > 0 else -progress
        return max(progress_reward, 0)
