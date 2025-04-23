import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces rewards based on strategic positioning and adaptive movement for defensive resilience."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Calculates new reward based on strategic defensive positions and quick transitions to counter-attack."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        tactical_reward = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            player_position = o['right_team'][o['active']]
            ball_position = o['ball'][:2]
            goal_position = [1, 0]  # right team's goal position

            # Positioning reward: Encourage maintaining a strategic depth, here set arbitrarily
            if player_position[0] < -0.5:
                tactical_reward[rew_index] = 0.05  # rewarded for strategic positioning behind mid-line

            # Transition reward: Encourage quick transition towards the ball if lost
            if o['ball_owned_team'] != 1 and np.linalg.norm(np.subtract(player_position, ball_position)) < 0.3:
                tactical_reward[rew_index] += 0.1  # quick adaptation to ball possession change

            reward[rew_index] += tactical_reward[rew_index]*3

        components["tactical_reward"] = tactical_reward
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
