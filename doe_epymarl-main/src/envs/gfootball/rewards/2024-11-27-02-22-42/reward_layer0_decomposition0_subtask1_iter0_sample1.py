import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tailored defensive skill reinforcement reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_interceptions = 0
        self.sprint_actions = 0
        self.stop_sprint_actions = 0
        self.pass_actions = 0

    def reset(self):
        self.ball_interceptions = 0
        self.sprint_actions = 0
        self.stop_sprint_actions = 0
        self.pass_actions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['defensive_metrics'] = {
            'ball_interceptions': self.ball_interceptions,
            'sprint_actions': self.sprint_actions,
            'stop_sprint_actions': self.stop_sprint_actions,
            'pass_actions': self.pass_actions
        }
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        defensive_metrics = from_pickle['defensive_metrics']
        self.ball_interceptions = defensive_metrics['ball_interceptions']
        self.sprint_actions = defensive_metrics['sprint_actions']
        self.stop_sprint_actions = defensive_metrics['stop_sprint_actions']
        self.pass_actions = defensive_metrics['pass_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_action_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            player_obs = observation[idx]

            # Encourage ball interception
            if player_obs['ball_owned_team'] == 1:  # Our team owns the ball
                self.ball_interceptions += 1
                components["defensive_action_reward"][idx] += 0.1

            # Encourage sprint and stop sprint based on position and game dynamics
            if player_obs['sticky_actions'][7]:  # Sprinting is action index 7
                self.sprint_actions += 1
                components["defensive_action_reward"][idx] += 0.05

            if player_obs['sticky_actions'][8]:  # Stop sprint is action index 8
                self.stop_sprint_actions += 1
                components["defensive_action_reward"][idx] += 0.05

            # Encourage passing from defense to midfield or forwards
            if player_obs['game_mode'] == 0 and player_obs['ball_owned_player'] == idx:
                self.pass_actions += 1
                components["defensive_action_reward"][idx] += 0.1

            # Combine the rewards
            reward[idx] += components["defensive_action_reward"][idx]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
