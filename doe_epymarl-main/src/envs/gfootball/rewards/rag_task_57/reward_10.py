import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized reward focused on offensive strategies 
    involving midfielders and strikers for ball delivery and play finishing."""

    def __init__(self, env):
        super().__init__(env)
        self.play_progress = {}
        self.midfield_success_rate = 0.1  # Incentivizing midfielders to pass successfully towards the strikers
        self.striker_success_rate = 0.5  # Higher incentive for strikers to score
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.play_progress = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.play_progress
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.play_progress = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Setup for reward components
            components.setdefault('midfield_contribution', [0.0] * len(reward))
            components.setdefault('striker_contribution', [0.0] * len(reward))
            
            # Check scenario for midfielders passing to strikers as a successful tactical play
            if o['left_team_roles'][o['active']] in (4, 5, 6, 7):  # Assuming these roles are midfielders
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    role_of_ball_receiver = observation[(rew_index + 1) % len(reward)]['left_team_roles'][observation[
                        (rew_index + 1) % len(reward)]['active']]
                    if role_of_ball_receiver in (8, 9):  # Assuming these roles are strikers
                        components['midfield_contribution'][rew_index] = self.midfield_success_rate
                        reward[rew_index] += components['midfield_contribution'][rew_index]
                        self.play_progress.setdefault(rew_index, []).append('midfield_play')

            # Reward scenario for strikers finishing the play
            if o['left_team_roles'][o['active']] in (8, 9):  # Strikers roles
                if reward[rew_index] > 0:  # If a goal is scored
                    components['striker_contribution'][rew_index] = self.striker_success_rate
                    reward[rew_index] += components['striker_contribution'][rew_index]
                    self.play_progress.setdefault(rew_index, []).append('striker_finish')

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
