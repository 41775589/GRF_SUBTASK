import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized dribbling and sprint reward."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the internal sticky action counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the current state of the environment including wrapper state."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserializes the state of the environment including wrapper state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modifies the reward based on dribbling actions and use of sprint."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribble_sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, obs in enumerate(observation):
            # Encourage dribbling with the ball and using sprint
            dribbling = obs['sticky_actions'][9]  # assuming index for dribble action is 9
            sprinting = obs['sticky_actions'][8]  # assuming index for sprint action is 8
            ball_owned = (obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active'])
             
            if ball_owned and dribbling and sprinting:
                components["dribble_sprint_reward"][rew_index] = 0.2  # reward for dribbling with sprint
            
            # Update total reward
            reward[rew_index] += components["dribble_sprint_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Takes an action in the environment and processes the accompanying rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        # Add sticky actions to info
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
