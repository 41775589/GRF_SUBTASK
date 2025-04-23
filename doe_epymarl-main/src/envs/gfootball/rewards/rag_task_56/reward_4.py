import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards focusing on defensive training, especially for improving goalkeeper 
    shot-stopping abilities and defenders' tackling and ball retention."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_score_difference = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_score_difference = None
        return super().reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter,
                                                'last_score_difference': self.last_score_difference}
        return super().get_state(to_pickle)

    def set_state(self, state):
        from_pickle = super().set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.last_score_difference = from_pickle['last_score_difference']

    def reward(self, reward):
        """
        Redirects the game reward to emphasize defensive capabilities:
        - Decreases points for goals conceded
        - Increases points for successful tackles, interceptions and clearances by defenders and goalkeeper saves
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_save_reward": [0.0] * len(reward),
                      "defensive_action_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        current_score_diff = observation['score'][0] - observation['score'][1]
        if self.last_score_difference is not None:
            goal_difference = current_score_diff - self.last_score_difference
            
            if goal_difference > 0:  
                # Bonus for preventing goals (defensive success)
                for i in range(len(reward)):
                    components["goalkeeper_save_reward"][i] = 0.5
                    reward[i] += components["goalkeeper_save_reward"][i]
            elif goal_difference < 0:  
                # Penalize for goals conceded
                for i in range(len(reward)):
                    reward[i] -= 1

        self.last_score_difference = current_score_diff

        # Checking for defensive actions: tackles, clearances, and interceptions
        for i, agent_observation in enumerate(observation):
            if agent_observation.get('ball_owned_team', -1) == 0 and agent_observation['left_team_roles'][agent_observation['active']] in [1, 2, 3, 4]:
                # Reward player for having the ball in a defensive role
                components["defensive_action_reward"][i] = 0.3
                reward[i] += components["defensive_action_reward"][i]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_active
        
        return observation, reward, done, info
