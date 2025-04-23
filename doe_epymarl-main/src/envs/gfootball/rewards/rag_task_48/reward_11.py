import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for executing effective high passes in midfield aiming to set up direct scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.high_pass_reward = 0.3
        self.scoring_opportunity_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the state of the environment and the sticky action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the current state of the environment and the wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment and the wrapper from a pickled state.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on the execution of high passes from midfield and creating scoring opportunities.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "scoring_opportunity_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation), "Length of reward and observation must match."
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][0]
            active_player_pos = o['left_team'][o['active']]
            
            # Check if the controlled player is in midfield and whether high pass is performed
            is_midfield = -0.3 <= active_player_pos[0] <= 0.3  # Simple midfield detection
            high_pass_action = o['sticky_actions'][9]  # Assuming index 9 denotes high pass action
            
            # Scoring opportunity defined by ball landing in attacking third after high pass
            scoring_opportunity_zone = 0.6 <= ball_pos
            
            if high_pass_action:
                if is_midfield:
                    components["high_pass_reward"][rew_index] = self.high_pass_reward
                    
                if scoring_opportunity_zone:
                    components["scoring_opportunity_reward"][rew_index] = self.scoring_opportunity_reward
            
            # Update the reward for this player
            reward[rew_index] += (components["high_pass_reward"][rew_index] + 
                                components["scoring_opportunity_reward"][rew_index])
        
        return reward, components

    def step(self, action):
        """
        Executes a step in the environment, computes the reward, and returns it with additional info.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        new_observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in new_observation:
            if 'sticky_actions' in agent_obs:
                for i, active in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += int(active)

        return observation, reward, done, info
