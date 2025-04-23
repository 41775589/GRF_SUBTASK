import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized goalkeeper training reward system."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize the reward for shot stopping and starting counter-attacks
        self.shot_stopping_reward = 0.5
        self.passing_reward = 0.3
        self.quick_reflexes_reward_multiplier = 0.2
        self.max_goalkeeper_speed = 0.05
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
        components = {"base_score_reward": reward.copy(),
                      "shot_stopping_reward": [0.0, 0.0],
                      "passing_reward": [0.0, 0.0],
                      "quick_reflexes_reward": [0.0, 0.0]}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active player is the goalkeeper
            if o['active'] == 0 and o['left_team_roles'][o['active']] == 0:
                if o['ball_owned_team'] == 0 or (o['ball_owned_player'] == o['active']):
                    ball_speed = np.linalg.norm(o['ball_direction'])
                    
                    # Reward quick reflexes for saving fast moving shots
                    if ball_speed > self.max_goalkeeper_speed:
                        components["quick_reflexes_reward"][rew_index] = (ball_speed - self.max_goalkeeper_speed) * self.quick_reflexes_reward_multiplier
                        reward[rew_index] += components["quick_reflexes_reward"][rew_index]

                    # Introduce shot-stopping reward when the goalkeeper stops a fast ball
                    if o['ball_owned_team'] == 0:
                        components["shot_stopping_reward"][rew_index] = self.shot_stopping_reward
                        reward[rew_index] += components["shot_stopping_reward"][rew_index]

                # Reward the goalkeeper for initiating counter-attacks with accurate passes
                # Let's assume passing is considered "accurate" when the speed vector after passing increases significantly
                if 'ball_direction' in o and np.linalg.norm(o['ball_direction']) > 1.2 * self.max_goalkeeper_speed:
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

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
