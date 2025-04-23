import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on dribbling- and movement-related achievements."""
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_rewards = {}
        self.position_change_reward = 0.1
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_rewards.clear()
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['dribble_rewards'] = self.dribble_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_rewards = from_pickle['dribble_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_rewards": [0.0] * len(reward),
                      "position_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for starting or performing a dribble.
            dribble_action = o['sticky_actions'][9]  # action_dribble is the 10th element
            if dribble_action == 1:
                # Update dribble_reward_counter if not already started
                if not self.dribble_rewards.get(rew_index, False):
                    components["dribble_rewards"][rew_index] = 0.05
                    self.dribble_rewards[rew_index] = True
            
            # Reward for movement changes while dribbling.
            if self.dribble_rewards.get(rew_index, False):
                current_pos = np.array(o['left_team'][o['active']])
                previous_pos = np.array(o['left_team'][o['active']]) - np.array(o['left_team_direction'][o['active']])
                dist_moved = np.linalg.norm(current_pos - previous_pos)
                components["position_change_reward"][rew_index] = dist_moved * self.position_change_reward
                
            # Accumulating rewards.
            reward[rew_index] += components["dribble_rewards"][rew_index] + components["position_change_reward"][rew_index]
                
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
                self.sticky_actions_counter[i] += action
        info.update({f"sticky_actions_{i}": self.sticky_actions_counter[i] for i in range(10)})
        return observation, reward, done, info
