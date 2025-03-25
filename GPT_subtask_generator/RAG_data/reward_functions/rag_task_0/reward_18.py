import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive strategies like accurate shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state including the sticky action counter."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state including the sticky action counter."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Calculate the enhanced reward based on shooting, dribbling and passing skills."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            # Shooting reward: successful goal attempts
            if o['score'][1] > o['score'][0]:  # assuming agent is on right team, index 1
                components["shooting_reward"][rew_index] = 1.0  # Suppose scoring a goal is a +1 reward

            # Dribbling reward: consider dribbling successful if player retains possession while moving
            if o['ball_owned_player'] == o['active'] and np.any(o['sticky_actions'][8:10]):  # sprint or dribble action
                components["dribbling_reward"][rew_index] = 0.1  # Dribbling reward

            # Passing reward: pass actions that change ball possession to a teammate, model simplistically
            if 'action' in o and (o['action'] in [football_action_set.action_short_pass, football_action_set.action_long_pass]):
                components["passing_reward"][rew_index] = 0.05  # Passing reward

            # Summing up all the rewards for particular agent
            reward[rew_index] += sum(components[component][rew_index] for component in components)

        return reward, components

    def step(self, action):
        """Step through the environment, apply rewards, and return information."""
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
