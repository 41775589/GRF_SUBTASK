import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward based on offensive skills execution accuracy, 
    including passing, shooting, dribbling, and creating scoring opportunities.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        
        # Initialize reward components weights
        self.pass_reward_weight = 0.05
        self.shot_reward_weight = 0.3
        self.dribble_reward_weight = 0.1
        self.sprint_reward_weight = 0.02
        
        # Initialize action counters and sticky actions
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
        components = {"base_score_reward": reward.copy(), "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward), "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            actions = o['sticky_actions'][0:10]
            if actions[8]:  # Action dribble
                components["dribble_reward"][rew_index] = self.dribble_reward_weight
            if actions[9]:  # Action sprint
                components["sprint_reward"][rew_index] = self.sprint_reward_weight

            # Assuming the following hypothetical mapping to rewarded actions
            if 'game_mode' in o:
                if o['game_mode'] == 6:  # Position for shot (penalty)
                    components["shot_reward"][rew_index] = self.shot_reward_weight
            
                if o['game_mode'] == 3:  # Position for pass (free kick)
                    components["pass_reward"][rew_index] = self.pass_reward_weight
            
            # Aggregate computed components to the base reward
            reward[rew_index] += (components["pass_reward"][rew_index] +
                                  components["shot_reward"][rew_index] +
                                  components["dribble_reward"][rew_index] +
                                  components["sprint_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding info about reward components for analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
