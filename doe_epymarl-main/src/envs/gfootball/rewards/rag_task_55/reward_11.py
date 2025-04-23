import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive tactics, particularly sliding and standing tackles."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize a counter for sticky actions, primarily for tackling
        self.sticky_actions_counter = np.zeros(10, dtype=int) 
        
    def reset(self):
        # Reset the sticky actions counter on environment reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def step(self, action):
        # Perform a step in the environment
        observation, reward, done, info = self.env.step(action)
        
        # Process the reward with the specialized reward function
        reward, components = self.reward(reward)
        
        # Record the final reward and its components in the info dictionary
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)

        # Update the sticky actions counter based on observations
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # If the player has not successfully tackled, skip
            active = o.get("active", -1)
            if o['sticky_actions'][8] == 1:  # action_sprint
                components["tackle_reward"][rew_index] = -0.1
            if o['sticky_actions'][9] == 1:  # action_dribble
                components["tackle_reward"][rew_index] = -0.1

            # Calculate distance to the ball
            player_pos = o['right_team'][active] if o['ball_owned_team'] == 1 else o['left_team'][active]
            ball_pos = o['ball'][:2]
            distance_to_ball = np.sqrt(np.sum((ball_pos - player_pos)**2))

            # Reward negative of distance if a tackle action (either standing or sliding) is performed
            if o['sticky_actions'][3] == 1 or o['sticky_actions'][8] == 1:  # action_right or action_sprint for tackle
                components["tackle_reward"][rew_index] += -distance_to_ball
            
            # Calculate the reward as a combination of base score reward and tackle reward
            reward[rew_index] = components["base_score_reward"][rew_index] + components["tackle_reward"][rew_index]
        
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle
