import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive transitions by evaluating stop-start behavior."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the idle and sprint action indices based on the specific environment setup
        # Assuming 9 corresponds to 'action_sprint' and 0 corresponds to 'action_idle'
        self.idle_action_idx = 0
        self.sprint_action_idx = 9
        self.prev_actions = None

    def reset(self):
        """Reset the environment and tracking variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_actions = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Extract state for saving."""
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from loaded data."""
        from_pickle = self.env.set_state(state)
        data = from_pickle.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = np.array(data.get("sticky_actions_counter", []), dtype=int)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on defensive transitions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            actions = o.get('sticky_actions', [])

            if self.prev_actions is not None:
                # Check if the player stopped running instantly in the last step
                was_sprinting = self.prev_actions[self.sprint_action_idx] == 1
                is_idle_now = actions[self.idle_action_idx] == 1

                # Check if the player started sprinting instantly from an idle state
                was_idle = self.prev_actions[self.idle_action_idx] == 1
                is_sprinting_now = actions[self.sprint_action_idx] == 1

                if (was_sprinting and is_idle_now) or (was_idle and is_sprinting_now):
                    transition_reward = 0.1
                    reward[rew_index] += transition_reward
                    components["transition_reward"][rew_index] = transition_reward

            # Update the previous actions for the next step
            self.prev_actions = actions

        return reward, components

    def step(self, action):
        """Perform a step in the environment, apply the reward wrapper, and return results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        return observation, reward, done, info
