import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that reinforces effective mid to long-range passing in a cooperative team play setting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.3  # Reward increment for successful long passes
        self.pass_threshold = 0.3  # Minimum distance to define a long pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        from_pickle = self.env.set_state(from_pickle)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_player = o['active']
            team_positions = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
            player_position = team_positions[current_player]

            # Iterate through all team members to check for long passes.
            for teammate_index, teammate_position in enumerate(team_positions):
                if teammate_index != current_player:
                    distance = np.linalg.norm(teammate_position - player_position)
                    
                    # Check if the distance qualifies as a long pass and if a long pass is active.
                    if distance > self.pass_threshold and self.sticky_actions_counter[-1]:  # Last action of sticky_actions is 'long pass'
                        # Reward for making a strategic long pass
                        components["pass_reward"][rew_index] = self.pass_reward
                        reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Incorporate calculated rewards and other details into 'info' to track metrics
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter if a sticky action is activated
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
