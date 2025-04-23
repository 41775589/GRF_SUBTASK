import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides additional rewards for executing high passes with precision,
    focusing on power assessment and trajectory control.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_power_threshold = 0.7  # Assuming normalized power range [0, 1], high passes typically need higher power.
        self.z_threshold = 0.15  # A threshold to determine if the pass is high based on the z component of ball direction.
        self.high_pass_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Include any stateful elements from the wrapper to pickle when saving the state
        # Currently, there are no extra stateful elements.
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Unpack any state elements for this wrapper before passing the state back to the environment.
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Augments the base reward with additional reward for successfully executing high passes.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if o['game_mode'] in [2, 3] and o['ball_owned_team'] == 1:  # FreeKick or GoalKick game modes assuming right team.
                ball_dir = o['ball_direction']
                if ball_dir[2] > self.z_threshold:
                    # Check if high pass (based on z component of direction)
                    action_set = o['sticky_actions']
                    if action_set[9] == 1:  # Assuming action index 9 corresponds to the 'high pass' action
                        components['high_pass_reward'][rew_index] = self.high_pass_reward
                        reward[rew_index] += components['high_pass_reward'][rew_index]

        return reward, components

    def step(self, action):
        """
        Uses the defined reward system to augment rewards at each step.
        Also includes tracking of sticky actions and returning detailed info dictionary.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
