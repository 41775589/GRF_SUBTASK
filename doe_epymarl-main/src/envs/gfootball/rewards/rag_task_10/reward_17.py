import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive actions."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_actions_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            ball_owner = o['ball_owned_team']
            ball_owner_player = o['ball_owned_player']
            acting_team = (0 if o['right_team_active'].all() else 1)
            active_player = o['active']
            
            # Increase the reward for defensive actions when the other team owns the ball
            if ball_owner != -1 and ball_owner != acting_team:
                # Check for sliding, stop-dribble, stop-movement via sticky actions
                if o['sticky_actions'][9]:  # Action ID for sliding
                    components["defensive_actions_reward"][rew_index] += 0.1
                if o['sticky_actions'][8]:  # Action ID for dribble which might indicate stopping others' dribble
                    components["defensive_actions_reward"][rew_index] += 0.05
                if o['sticky_actions'][0] == 0 and o['sticky_actions'][4] == 0:  # No left/right movement
                    components["defensive_actions_reward"][rew_index] += 0.03
            
            # Cumulative Reward Adjustment
            reward[rew_index] += components["defensive_actions_reward"][rew_index]

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
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_value
                info[f"sticky_actions_{i}"] = action_value
        return observation, reward, done, info
