import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles, focusing on the timing and precision of these moves under high-pressure situations."""

    def __init__(self, env):
        super().__init__(env)
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
                      "tackle_timing_precision": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            components["tackle_timing_precision"][rew_index] = 0.0
            
            # Checking if it's a high-pressure situation (oppositions are close and moving towards the player)
            if obs['ball_owned_team'] == 0:  # Assuming our agent is in team 0
                opponents = obs['right_team']
                teammates = obs['left_team']
                ball_position = obs['ball']
                player_position = teammates[obs['active']]

                close_opponents = np.sum(np.linalg.norm(opponents - player_position, axis=1) < 0.1)
                if close_opponents > 2:  # If more than two opponents are close
                    # Check if tackle was successful
                    if 'action' in obs and obs['sticky_actions'][9] == 1:  # Assuming action index 9 is for tackling
                        components["tackle_timing_precision"][rew_index] = 1.0 * close_opponents
                    
            # Aggregate the crafted rewards with base rewards
            reward[rew_index] += components["tackle_timing_precision"][rew_index]
        
        return reward, components

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)
        rewards, components = self.reward(rewards)
        info["final_reward"] = sum(rewards)
        
        # Updating info with components summary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observations:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        
        return observations, rewards, done, info
