import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive strategies including shooting, dribbling, and passing."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._passing_coefficient = 0.2
        self._shooting_coefficient = 0.3
        self._dribbling_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_component": [0.0] * len(reward),
                      "shooting_component": [0.0] * len(reward),
                      "dribbling_component": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assume 'designated' refers to the player leading the attack
            active_player = o['designated']
            
            # Add rewards for dribbling: check if 'action_dribble' (index 9) is active
            if o['sticky_actions'][9] == 1:
                components["dribbling_component"][rew_index] += self._dribbling_coefficient

            # Add rewards for successful passes, comparing active and designated player
            if o['ball_owned_player'] == active_player and o['ball_owned_team'] in [0, 1]:
                # simulate a successful pass being a change in ball ownership after the pass command
                last_action_was_pass = np.any(o['sticky_actions'][5:7]) # considering action_bottom_right, and action_bottom_left
                if last_action_was_pass:
                    components["passing_component"][rew_index] += self._passing_coefficient
            
            # Reward for shooting at goal: assume shooting when near the opponent's goal area
            if np.abs(o['ball'][0]) > 0.9 and np.abs(o['ball'][1]) < 0.044:
                components["shooting_component"][rew_index] += self._shooting_coefficient

            # Aggregate all individual components to form the total reward
            reward[rew_index] = 1 * components["base_score_reward"][rew_index] \
                                + components["passing_component"][rew_index] \
                                + components["shooting_component"][rew_index] \
                                + components["dribbling_component"][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
