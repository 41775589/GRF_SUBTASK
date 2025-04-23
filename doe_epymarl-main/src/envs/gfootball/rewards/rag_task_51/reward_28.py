import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward function for specialized goalkeeper training."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_training_reward = 0.05

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
        # Get the current observations
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "shot_stopping_reward": [0.0] * len(reward),
                      "quick_reflexes_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Processing the reward for each agent (specifically focusing on the goalkeeper)
        for rew_index, r in enumerate(reward):
            o = observation[rew_index]

            if 'active' in o:
                # Identifying the goalkeeper
                if o['active'] == o['left_team_roles'][0] or o['active'] == o['right_team_roles'][0]:
                    # Reward for shot-stopping: enhance if the agent stops shots when the ball is very close to the goal
                    if 'ball_owned_team' in o and o['ball_owned_team'] != o['ball_owned_player'] and \
                            abs(o['ball'][0]) > 0.8 and abs(o['ball'][1]) < 0.044:
                        components["shot_stopping_reward"][rew_index] = self.goalkeeper_training_reward
                        reward[rew_index] += components["shot_stopping_reward"][rew_index]

                    # Reward for quick reflexes: based on the quick change in ball direction close to the goal area
                    if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 1.0 and \
                            abs(o['ball'][0]) > 0.9:
                        components["quick_reflexes_reward"][rew_index] = self.goalkeeper_training_reward
                        reward[rew_index] += components["quick_reflexes_reward"][rew_index]

                    # Reward for initiating a counter-attack: if the goalkeeper has the ball and makes a pass
                    if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'ball_owned_player' in o and \
                            o['ball_owned_player'] == o['active'] and any(o['sticky_actions'][5:7]):
                        components["counter_attack_reward"][rew_index] = self.goalkeeper_training_reward
                        reward[rew_index] += components["counter_attack_reward"][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
