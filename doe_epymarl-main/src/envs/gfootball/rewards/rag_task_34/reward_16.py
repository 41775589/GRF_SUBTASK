import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized reward for close-range attacks against the goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions

        # Reward for successful dribbling or shooting close to goal within goal area (x>-0.8)
        self.goal_area_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper_sticky_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_area_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Determine if the ball is in close proximity to the goalkeeper and owned by player's controlled player
            if o['ball'][0] > -0.8 and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Check for a shot or dribble action
                active_actions = o['sticky_actions'][8:]  # Action indices 8 and 9 are sprint and dribble, respectively
                if active_actions.sum() > 0:  # If either sprint or dribble is active
                    components["goal_area_reward"][rew_index] = self.goal_area_reward
                    reward[rew_index] += self.goal_area_reward
                    
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
