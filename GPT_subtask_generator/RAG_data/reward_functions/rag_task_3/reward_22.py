import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting training with accuracy and power under pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._shot_accuracy_counter = {}
        self._default_shot_reward = 0.5
        self._pressure_coef = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._shot_accuracy_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shot_accuracy_counter'] = self._shot_accuracy_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._shot_accuracy_counter = from_pickle['shot_accuracy_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = np.array(o['ball'][:2])
            goal_pos = np.array([1.0, 0.0])  # Simulating goal position on right

            if 'sticky_actions' in o:
                # Check if the player is attempting a shot.
                if o['sticky_actions'][9]:  # Index 9 corresponds to 'action_shot'
                    distance_to_goal = np.linalg.norm(ball_pos - goal_pos)
                    accuracy_reward = max(0, self._default_shot_reward - distance_to_goal)

                    # Pressure factor increases the reward if more defensive players are close
                    defense_presence = len([p for p in o['right_team'] if np.linalg.norm(p - ball_pos) < 0.1])
                    pressure_reward = self._pressure_coef * defense_presence

                    # Calculate total shot reward
                    total_shot_reward = accuracy_reward + pressure_reward
                    components['shot_accuracy_reward'][rew_index] = total_shot_reward
                    reward[rew_index] += total_shot_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
