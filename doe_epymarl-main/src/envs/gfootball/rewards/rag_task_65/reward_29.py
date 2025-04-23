import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on scenario-based actions, focusing on shooting and passing accuracy in game-like contexts."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_positions = []
        self.passing_counts = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.shooting_positions = []
        self.passing_counts = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['shooting_positions'] = self.shooting_positions
        state['passing_counts'] = self.passing_counts
        return state

    def set_state(self, state):
        state = self.env.set_state(state)
        self.shooting_positions = state['shooting_positions']
        self.passing_counts = state['passing_counts']
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy": [0.0] * len(reward),
                      "passing_effectiveness": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            ball_pos = obs['ball'][:2]
            shooting_range = 0.1
            passing_action_active = obs['sticky_actions'][9]

            # Shooting precision check
            # Mainly focuses within a close range to the goal (assuming goal at y=0, x=1 or x=-1 in normalized coords)
            if np.abs(ball_pos[0]) > 1 - shooting_range and np.abs(ball_pos[1]) < shooting_range:
                self.shooting_positions.append(ball_pos)
                if len(self.shooting_positions) > 1:  # Assuming more complexity in the trajectory can lead to higher rewards
                    components["shooting_accuracy"][idx] = 1.5
                else:
                    components["shooting_accuracy"][idx] = 1.0
                reward[idx] += components["shooting_accuracy"][idx]

            # Passing effectiveness check based on the number of passes executed with action 'dribble' active
            if passing_action_active:
                self.passing_counts.append(True)
                if len(self.passing_counts) % 5 == 0:  # Reward for every 5 successful passes
                    components["passing_effectiveness"][idx] = 0.3
                reward[idx] += components["passing_effectiveness"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        if obs is not None:
            for agent_obs in obs:
                for i, act in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = act
                    info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
