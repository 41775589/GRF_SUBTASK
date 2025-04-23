import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a scenario-based reward for shooting and passing precision."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define shooting and passing thresholds
        self.passing_threshold = 0.8
        self.shooting_threshold = 1.0
        self.shooting_reward = 2.0
        self.passing_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = ''
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        # Base score reward as initially received from the environment
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Implement logic for shooting reward
            if 'ball_owned_team' in o and o['ball_owned_team'] in [0, 1]:
                ball_dist_to_goal = np.abs(o['ball'][0] - 1) if o['ball_owned_team'] == 0 else np.abs(o['ball'][0] + 1)
                if ball_dist_to_goal < self.shooting_threshold:
                    if o['game_mode'] in [2, 6]:  # Game modes that represent shooting scenarios
                        components["shooting_reward"][rew_index] = self.shooting_reward
                        reward[rew_index] += components["shooting_reward"][rew_index]

            # Implement logic for passing reward
            # Assuming that a successful pass increases the nearest teammate's control over the ball
            if 'ball_owned_player' in o and o['ball_owned_player'] != -1:
                ball_control_changes = len([i for i in o['sticky_actions'] if i > 0])
                if ball_control_changes > 0 and np.linalg.norm(o['ball_direction'][:2]) > self.passing_threshold:
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Add component values and final reward value to the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update observation on sticky actions for each agent
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
