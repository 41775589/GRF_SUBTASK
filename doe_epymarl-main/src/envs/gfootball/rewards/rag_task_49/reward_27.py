import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for shooting accurately from central field positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_positions = np.linspace(-0.3, 0.3, 5)  # Centralized field areas for shooting
        self.positional_reward = 0.05  # Reward for reaching each checkpoint position
        self.goal_reward = 1  # Reward for scoring a goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'positional_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Ensure the length of reward array matches the number of players (agents)
        assert len(reward) == len(observation)

        for index in range(len(reward)):
            o = observation[index]
            ball_x = o['ball'][0]  # Get X coordinates of the ball
            shooting_zone_rewards = [self.positional_reward for pos in self.shooting_positions if abs(ball_x - pos) < 0.1]
            
            if sum(shooting_zone_rewards) > 0:
                components['positional_reward'][index] = sum(shooting_zone_rewards)
                reward[index] += components['positional_reward'][index]

            # Additional reward for goal
            if o['score'][0] > 0 or o['score'][1] > 0:  # Assuming a goal is scored
                reward[index] += self.goal_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        cumulative_reward, components = self.reward(reward)
        info['final_reward'] = sum(cumulative_reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f'sticky_actions_{i}'] = self.sticky_actions_counter[i]
        return observation, cumulative_reward, done, info
