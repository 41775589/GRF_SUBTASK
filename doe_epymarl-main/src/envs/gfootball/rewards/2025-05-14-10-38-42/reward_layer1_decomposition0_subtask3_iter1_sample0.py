import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on improving shooting and dribbling skills in offensive plays, contributing to scoring efficiency."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions
        self.shooting_reward_multiplier = 2.0  # Increase shooting reward factor
        self.dribbling_reward_multiplier = 1.5  # Increase dribbling reward factor

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Customize rewards based on specific actions: shooting and dribbling in advantageous positions."""
        observation = self.env.unwrapped.observation()
        # Components for each type of reward calculation
        components = {"base_score_reward": reward.copy(), "shooting_reward": [0.0] * len(reward), "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Checking if the active player owns the ball
            if o.get('ball_owned_player') == o.get('active') and o.get('ball_owned_team') == 1: # Assuming team 1 is the controlled team
                # Dribble action increment
                if 'sticky_actions' in o and o['sticky_actions'][8]:  # Assuming index 8 is dribble
                    components["dribbling_reward"][i] = self.dribbling_reward_multiplier
                    reward[i] += components["dribbling_reward"][i]

                # Shooting rewards when in the proximity of the goal
                goal_distance = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5  # Assuming shooting towards right goal
                if 'sticky_actions' in o and (o['sticky_actions'][6] or o['sticky_actions'][5]) and goal_distance < 0.2:  # Assuming indexes 6, 5 are shooting actions
                    components["shooting_reward"][i] = self.shooting_reward_multiplier
                    reward[i] += components["shooting_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Collect and display final reward and individual component values
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions status
        self.sticky_actions_counter.fill(0)
        if observation:
            for idx, act in enumerate(observation[0]['sticky_actions']):
                if act == 1:
                    self.sticky_actions_counter[idx] += 1
                info[f"sticky_actions_{idx}"] = self.sticky_actions_counter[idx]

        return observation, reward, done, info
