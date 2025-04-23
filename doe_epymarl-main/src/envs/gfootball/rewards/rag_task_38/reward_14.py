import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on efficient transitions from defense to attack,
    emphasizing long passes and quick transitions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For tracking sticky actions

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the wrapper state to a pickle object.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state from the pickle object.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward function to promote quick transitions and long passes
        from defense to attack.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Calculate transition reward based on position change and ball control
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # Assuming agent's team is 1
                original_pos = o.get('left_team', [])[o['active']]
                new_pos = original_pos + o.get('left_team_direction', [])[o['active']]
                dist_moved = np.linalg.norm(new_pos - original_pos)

                # Reward long passes (simulated by checking the active player)
                if 'action_long_pass' in o['sticky_actions']:
                    components['transition_reward'][rew_index] = 0.2

                # Reward for moving forward quickly towards the attacking side
                if new_pos[0] > original_pos[0] and dist_moved > 0.05:  # Threshold to consider as 'quick'
                    components['transition_reward'][rew_index] += 0.1 * dist_moved

                # Finally update the reward array
                reward[rew_index] += components['transition_reward'][rew_index]

        return reward, components

    def step(self, action):
        """
        Step through the environment and augment rewards data with component values.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
