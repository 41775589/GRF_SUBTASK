import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for synergizing defensive roles."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        self.num_defensive_positions = 5
        self.defensive_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['position_rewards'] = self.position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards = from_pickle.get('position_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_position_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            # Each agent gets a bonus for taking up a close, yet distinct position to the defensive goal area
            # when game_mode signifies a high-pressure scenario, such as corners or penalties (game_mode > 2).
            if o['game_mode'] > 2:
                x, y = o['left_team'][o['active']]
                # Calculate proximity to the goal area (-1, y with |y| <=0.2)
                proximity = abs(x + 1)  # Negative x, ideal position is near -1
                diversity_bonus = len(set([(round(player[0], 1), round(player[1], 1)) for player in o['left_team']]))
                uniq_position = (round(x, 1), round(y, 1))
                
                if uniq_position not in self.position_rewards:
                    self.position_rewards[uniq_position] = 1
                    position_value = self.defensive_reward * (1 - proximity) * (diversity_bonus / len(o['left_team']))
                    components['defensive_position_reward'][rew_index] = position_value
                    reward[rew_index] += position_value

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Include the reward components for diagnosis or debugging purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
