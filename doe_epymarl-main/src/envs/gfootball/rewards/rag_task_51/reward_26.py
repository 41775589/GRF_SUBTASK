import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on goalkeeper training."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward factors for different goalkeeper actions
        self.save_reward = 0.3
        self.reflex_reward = 0.2
        self.passing_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No state to restore in this simple example
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_reward": [0.0] * len(reward),
                      "reflex_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
            
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_role = o['left_team_roles'][o['active']] if o['ball_owned_team'] == 0 else o['right_team_roles'][o['active']]

            # Give additional reward if the goalie (role index 0) stops a goal or clears the ball
            if active_player_role == 0 and o['ball_owned_player'] == o['active']:
                components["save_reward"][rew_index] = self.save_reward
                reward[rew_index] += components["save_reward"][rew_index]

            # Reward quick reflex actions in changes of the ball's direction
            if np.linalg.norm(o['ball_direction'][:2]) > 0.5:
                components["reflex_reward"][rew_index] = self.reflex_reward
                reward[rew_index] += components["reflex_reward"][rew_index]

            # Reward accurate passes to initiate counter-attacks
            if o['ball_owned_player'] == o['active'] and 'action' in o and o['action'] in [7,10]:  # Actions 7 & 10 can denote passing actions
                components["passing_reward"][rew_index] = self.passing_reward
                reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
