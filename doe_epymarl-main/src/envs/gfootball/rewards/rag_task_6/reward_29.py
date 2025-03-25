import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards efficient stamina management in Google Research Football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the number of checkpoints for stamina management.
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.05

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {"base_score_reward": reward}

        sticky_actions = [
            'action_sprint',
            'action_idle'
        ]

        for o, rew in zip(observation, reward):
            sticky_states = o['sticky_actions']

            # Determine if sprint is active
            sprint_active = sticky_states[8] == 1  # index 8 is 'action_sprint'
            idle_active = sticky_states[0] == 0 and np.all(sticky_states[1:8] == 0)  # indices 1-8 represent movement

            stamina_state = o['right_team_tired_factor' if o['ball_owned_team'] == 1 else 'left_team_tired_factor']
            stamina_penalty = 1.0 - np.mean(stamina_state)

            # Only give reward if the player is minimizing stamina usage effectively
            if sprint_active and idle_active:
                # Apply more complex condition for rewards related to stamina conservation
                checkpoint_index = int(stamina_penalty * self._num_checkpoints)

                # Only reward once per stamina level threshold crossed
                if checkpoint_index not in self._collected_checkpoints:
                    self._collected_checkpoints[checkpoint_index] = True
                    rew += self._checkpoint_reward
                    components['stamina_management_reward'] = self._checkpoint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info['final_reward'] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, new_reward, done, info
