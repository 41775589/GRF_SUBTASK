import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that encourages efficient sprint and movement behavior,
    focusing on controlling stamina and positional integrity.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the sticky actions counter upon environment reset.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state including sticky actions counter.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state including sticky actions counter.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward function to include penalties or bonuses for specific sticky action usage, aiming at stamina conservation.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_penalty": [0.0] * len(reward),
                      "stop_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Encourage stopping sprint when not necessary.
            sprint_action_index = 8  # assuming 8 is the sprint action index
            move_actions_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # assuming these are the indices for moving actions
            sprint_usage = obs['sticky_actions'][sprint_action_index]
            
            if sprint_usage:
                components["sprint_penalty"][rew_index] = -0.01
            else:
                components["stop_bonus"][rew_index] = 0.02

            # Tracking usage of move actions with sprint
            if any(obs['sticky_actions'][index] for index in move_actions_indices):
                if sprint_usage:
                    components["sprint_penalty"][rew_index] -= 0.01

            # Calculate the final adjusted reward
            reward[rew_index] += components["sprint_penalty"][rew_index] + components["stop_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take an action using the environment's step function and augment the returned reward and info using the adjusted reward schema.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
