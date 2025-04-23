import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on ball control, space exploitation, and strategic play."""
    
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
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_reward": [0.0] * len(reward),
                      "space_exploitation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Apply rewards based on ball control, space utilization, and strategic movement
        for rew_index, o in enumerate(observation):
            # Ball control reward
            if o['ball_owned_team'] == 0:  # Assuming index 0 is the team of the agent
                components["control_reward"][rew_index] = 0.2  # Fixed small reward for ball possession
            
            # Space exploitation reward
            player_pos = o['left_team'][o['active']]
            team_pos = o['left_team']
            opponent_pos = o['right_team']
            team_center = np.mean(team_pos, axis=0)
            opponent_center = np.mean(opponent_pos, axis=0)
            distance_to_team_center = np.linalg.norm(player_pos - team_center)
            distance_to_opponent_center = np.linalg.norm(player_pos - opponent_center)
            
            # Reward players for moving into space away from their team center and close to the opponent's goal
            if distance_to_team_center > 0.2 and o['ball'][0] > 0:  # Encourage forward movement when having ball
                components["space_exploitation_reward"][rew_index] = distance_to_team_center * 0.1

            # Aggregate rewards with base reward
            reward[rew_index] += (components["control_reward"][rew_index] +
                                  components["space_exploitation_reward"][rew_index])

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
