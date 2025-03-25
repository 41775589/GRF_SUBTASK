import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive maneuver reward focusing on sliding tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment."""
        to_pickle['EnvStates'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['EnvStates']
        return from_pickle

    def reward(self, reward):
        """Customize the reward function to promote effective defensive sliding tackles."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Extract relevant data
            has_ball = (o['ball_owned_team'] == 0) and (o['ball_owned_player'] == o['active'])
            tackle_action_active = o['sticky_actions'][9]  # assuming 9 corresponds to the sliding tackle action

            # Apply rewards based on defensive behavior
            if tackle_action_active and has_ball:
                reward[rew_index] += 0.5  # Give reward for successful slide tackle with ball possession
                components["defensive_reward"][rew_index] += 0.5
            
        return reward, components

    def step(self, action):
        """Step function to apply the new reward mechanism."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
