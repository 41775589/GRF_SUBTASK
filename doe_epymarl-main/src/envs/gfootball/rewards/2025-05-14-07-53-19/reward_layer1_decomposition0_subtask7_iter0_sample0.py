import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to enhance long pass and sprint training in football scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_checkpoints = 5  # Divide the pitch into 5 regions for sprints and long passes
        self._checkpoint_reward = 0.1  # Reward for reaching each checkpoint
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To count sticky actions usage

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_reward": [0.0] * len(reward),
            "long_pass_sprint_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the agent executed a long pass or sprint
            if o['sticky_actions'][8]:  # Sprint action index
                self.sticky_actions_counter[8] += 1
            if o['sticky_actions'][9]:  # Dribble action index, assuming long pass completion as dribble
                self.sticky_actions_counter[9] += 1

            # Calculate distance from current to goal
            ball_pos = np.array(o['ball'][:2])  # Only considering x, y
            goal_pos = np.array([1, 0]) if o['ball_owned_team'] == 0 else np.array([-1, 0])
            distance_to_goal = np.linalg.norm(ball_pos - goal_pos)

            # Checkpoint logic based on ball position and actions performed
            checkpoint_interval = 1 / self._num_checkpoints
            current_checkpoint = int(distance_to_goal // checkpoint_interval)

            if self.sticky_actions_counter[8] >= current_checkpoint or self.sticky_actions_counter[9] >= current_checkpoint:
                # Assuming making a long pass towards the goal or sprinting effectively towards it
                components['long_pass_sprint_reward'][rew_index] = self._checkpoint_reward * (self._num_checkpoints - current_checkpoint)
                reward[rew_index] += components['long_pass_sprint_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add component values to info for logging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Include sticky actions counter updates
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
