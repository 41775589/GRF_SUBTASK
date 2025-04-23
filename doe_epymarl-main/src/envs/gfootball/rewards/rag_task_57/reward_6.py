import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a structured reward encouraging midfielders in creating opportunities and strikers in finishing."""
    
    def __init__(self, env, midfield_reward_importance=0.5, striker_reward_importance=0.5):
        super().__init__(env)
        # Weighting importance of rewards related to different player roles
        self.midfield_reward_importance = midfield_reward_importance
        self.striker_reward_importance = striker_reward_importance
        # Register if certain mid-field plays or striker goal attempts have occurred
        self.midfield_plays = 0
        self.striker_attempts = 0
        # Setup to track sticky actions for debugging purposes
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.midfield_plays = 0
        self.striker_attempts = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_plays'] = self.midfield_plays
        to_pickle['striker_attempts'] = self.striker_attempts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_plays = from_pickle['midfield_plays']
        self.striker_attempts = from_pickle['striker_attempts']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        # Component dictionary to keep track of different rewards
        components = {
            "base_score_reward": reward.copy(),
            "midfield_play_reward": [0.0] * len(reward),
            "striker_attempt_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for idx, o in enumerate(observation):
            # Checking mid-field strategies
            if 'midfield' in o.get('left_team_roles', []) + o.get('right_team_roles', []):
                self.midfield_plays += 1
                components["midfield_play_reward"][idx] = self.midfield_reward_importance

            # Checking striker's scoring opportunities
            if o.get('score_opportunity') and 'striker' in o.get('left_team_roles', []) + o.get('right_team_roles', []):
                self.striker_attempts += 1
                components["striker_attempt_reward"][idx] = self.striker_reward_importance

            # Adjusting the reward according to the components
            reward[idx] += (components["midfield_play_reward"][idx] + components["striker_attempt_reward"][idx])
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Updating info with component contributions
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
