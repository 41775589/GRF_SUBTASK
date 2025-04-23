import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper for goalkeeper training focused on shot-stopping,
    quick reflexes, and initiating counter-attacks with accurate passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Store the count of save attempts by the goalie
        self.save_attempts = 0
        self.successful_passes = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.save_attempts = 0
        self.successful_passes = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['save_attempts'] = self.save_attempts
        to_pickle['successful_passes'] = self.successful_passes
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.save_attempts = from_pickle['save_attempts']
        self.successful_passes = from_pickle['successful_passes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] in [3, 6]:  # Consider only free kicks or penalties
                if o['ball_owned_team'] == 0:  # Our team, assuming goalie is from left team
                    if o['ball_owned_player'] == o['active']:
                        # Assume goalkeeper saving goals
                        components["save_reward"][rew_index] = 1.0

            # Checking for pass success by the goalie
            if o['ball_owned_team'] == 0 and (np.abs(o['ball'][0]) > 0.8):  # Ball in the goalie region after action
                if any(o['sticky_actions'][[0, 4]]):  # Action was a pass left or right
                    components["pass_reward"][rew_index] = 1.0

            # Calculate total rewards
            reward[rew_index] += components["save_reward"][rew_index] + components["pass_reward"][rew_index]
        
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
