import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper that adds a reward component for maintaining control of the ball with 
    passing and dribble actions, aiding the transition from defense to attack.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the sticky actions counter when the environment is reset.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Add the current state of sticky actions to the pickle.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Retrieve the state of sticky actions from the pickle.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Reward function augmentation for controlling the ball with passes and dribbles.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "passing_control_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Encourages passing (both short and long) and dribbling when in possession
            if obs.get('ball_owned_team') == 0 and obs.get('sticky_actions')[8] == 1:
                components["passing_control_reward"][rew_index] = 0.05
            if obs.get('ball_owned_team') == 0 and (obs.get('sticky_actions')[5] == 1 or obs.get('sticky_actions')[6] == 1):
                components["passing_control_reward"][rew_index] = 0.1

            # Accumulating rewards for maintaining ball possession with successful passes or dribbles
            reward[rew_index] += components["passing_control_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Step the environment and modify the output dictionary to include reward components.
        """
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
