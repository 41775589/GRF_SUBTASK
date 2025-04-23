import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to focus on enhancing shooting precision from closed angles and tight spaces."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._precision_checkpoints = 5
        self._accuracy_reward = 0.2

    def reset(self):
        """Resets the environment and reward wrapper state."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state of the environment for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the environment from the deserialized state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Custom reward focusing on shooting precision in tight spaces."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_pos = o['ball']
            is_goal_attempt = o['game_mode'] == 6  # Checking if it is a penalty mode, representing close-shot training
            
            # Reward for precision in shot from a challenging angle
            if is_goal_attempt and o['ball_owned_team'] == o['active']:
                goal_x = 1 if o['ball_owned_team'] == 1 else -1
                dist_to_goal = abs(goal_x - ball_pos[0])
                y_dist = abs(ball_pos[1])

                # Assuming tighter spaces and challenging angles near the goal center y range (-0.2 to 0.2)
                if dist_to_goal < 0.2 and y_dist < 0.2:
                    checkpoint_index = int((dist_to_goal + y_dist) * self._precision_checkpoints)
                    components["precision_reward"][rew_index] = (self._precision_checkpoints - checkpoint_index) * self._accuracy_reward
                    reward[rew_index] += components["precision_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Steps through the environment applying given actions."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return obs, reward, done, info
