import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successfully clearing the ball from defensive zones under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save extra states if necessary
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore extra states if necessary
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            components["clearance_reward"][idx] = 0.0

            # Checking whether it's the defensive team and under pressure
            if (o['ball_owned_team'] == 0 and 
                np.linalg.norm(o['ball'][:2]) < 0.2 and 
                o['game_mode'] == 0):
                # Checking players around the ball within a small radius
                enemy_proximity = np.any(np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1) < 0.1)

                if enemy_proximity:
                    # Encourage clearing the ball far from the current position
                    direction = np.array([o['ball_direction'][0], o['ball_direction'][1]])
                    distance_cleared = np.linalg.norm(direction)
                    components["clearance_reward"][idx] = 0.5 * distance_cleared
                    reward[idx] += components["clearance_reward"][idx]

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
