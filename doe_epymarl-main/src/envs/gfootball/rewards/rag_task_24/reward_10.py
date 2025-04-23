import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards based on the effectiveness of mid to long-range passes,
    encouraging strategic use of passing to maintain possession and advance towards the opponent's goal.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_threshold = 0.2  # Define threshold distance for considering a pass to be long-range
        self.pass_reward_coefficient = 0.5  # Reward coefficient for successful long-range passes
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for tracking sticky actions

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
        components = {"base_score_reward": reward.copy(), "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Assumed structure based on the provided format in the problem setup
        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Check the current game mode to ensure normal play
            if obs['game_mode'] != 0:
                continue

            # Implementing passing rewards based on pass distances and ball ownership transitions
            ball_pos_prev_step = obs['ball'] if obs['ball_owned_team'] == obs['active'] else None

            if ball_pos_prev_step is not None:
                for teammate in (obs['left_team'] if obs['ball_owned_team'] == 0 else obs['right_team']):
                    dist = np.linalg.norm(ball_pos_prev_step[:2] - teammate[:2])
                    # Check if pass is longer than threshold and if the player is in control
                    if dist >= self.pass_threshold:
                        components["passing_reward"][rew_index] += self.pass_reward_coefficient
                        reward[rew_index] += components["passing_reward"][rew_index]

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
