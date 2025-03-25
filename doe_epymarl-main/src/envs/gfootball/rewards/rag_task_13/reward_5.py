import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds a reward for effective defensive actions construed as 'stopping' opponent's progress. """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Ensure to reset state-specific data if any
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy()}
        modified_reward = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Ensure the player is active and relevant
            if o['active'] == -1:
                continue

            # Select player-role specific observations: Only consider defensive actions by defenders
            if o['left_team_roles'][o['active']] not in [1, 2, 3, 4]:  # Only considering center back, left/right back, defense midfield
                continue

            # Define defensive reward components:
            has_ball = o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']
            opponent_close = np.any([np.linalg.norm(np.array(o['right_team'][i][:2]) - o['left_team'][o['active']][:2]) < 0.1
                                     for i in range(len(o['right_team']))])

            # Penalize if opponent is close and agent loses the ball
            penalty = -0.1 if opponent_close and not has_ball else 0.0
            # Reward blocking or taking the ball from the opponent
            defensive_reward = 0.3 if opponent_close and has_ball else 0.0

            # Adjust the reward components
            components["defensive_action_reward"] = [defensive_reward] * len(reward)
            components["defensive_penalty"] = [penalty] * len(reward)

            # Calculate new reward for this player
            modified_reward[rew_index] = reward[rew_index] + defensive_reward + penalty

        return modified_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
