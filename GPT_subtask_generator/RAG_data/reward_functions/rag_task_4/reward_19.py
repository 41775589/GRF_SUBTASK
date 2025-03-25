import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for advanced dribbling and sprinting in offensive play."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward increments for dribbling with sprint actions.
        self.dribble_sprint_checkpoint_reward = 0.2
        
    def reset(self):
        """Reset the wrap for a new environment episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the current state of the wrapper."""
        to_pickle['CheckpointRewardWrapper_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize and set the state of the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_actions_counter', 
                                                     np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Custom reward function to promote dribbling and sprinting during offensive plays."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Reward for dribbling with a sprint.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            has_ball = o['ball_owned_player'] == o['active'] and o['ball_owned_team'] in [0, 1]
            is_dribbling = o['sticky_actions'][9]
            is_sprinting = o['sticky_actions'][8]
            
            if has_ball and is_dribbling and is_sprinting:
                components["dribble_sprint_reward"][rew_index] = self.dribble_sprint_checkpoint_reward
                reward[rew_index] += components["dribble_sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step within the environment while managing internal state for reward calculation."""
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
        return observation, reward, done, info
