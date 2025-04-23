import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards the agent for precise stopping and sprinting in response to quick changes in gameplay.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward for mastering the stop-move techniques: stop-sprint and stop-moving commands reactively.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        move_rewards = [0.0] * len(reward)  # Initialize a move rewards list matching the `reward` dimension.
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_actions = o["sticky_actions"]
            
            # Check for stopping (action indices could depend on your environment's specific action mapping)
            if active_actions[0] == 0 and active_actions[4] == 0:  # Assuming actions 0 and 4 as possible directional stops
                if o['ball_owned_team'] == o['designated']:
                    # Adding small positive reward for effective stop after sprint or move
                    move_rewards[rew_index] += 0.1

            # Check if sprint was engaged directly after a stop
            if active_actions[8] == 1:  # Assuming action 8 is sprint
                if self.sticky_actions_counter[rew_index] < 2:  # Check if recently stopped
                    move_rewards[rew_index] += 0.2
                self.sticky_actions_counter[rew_index] += 1
            else:
                self.sticky_actions_counter[rew_index] = 0  # Reset on non-sprint action
            
            reward[rew_index] += move_rewards[rew_index]
        
        components["advanced_movement_reward"] = move_rewards
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
