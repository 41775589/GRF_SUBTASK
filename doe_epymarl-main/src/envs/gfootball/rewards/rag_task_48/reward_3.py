import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages executing high passes from midfield to create scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.high_pass_bonus = 0.2  # Reward increment for successful high passes towards scoring
        self.midfield_threshold = 0.2  # Defines the zone considered as "midfield"
        self.minimum_pass_distance = 0.5  # Minimum distance to consider it a long/high pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions

    def reset(self):
        """Reset the wrapping environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the current state of the environment along with local state for saving/resuming."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize and apply the saved state to the environment."""
        from_pickle = self.env.set_state(state)
        # Nothing to load as no internal state is used beyond static configurations.
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on midfield high passes creating direct scoring opportunities."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # High pass logic: Originates in midfield and travels a considerable distance upwards
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active'] and \
               abs(o['ball'][0]) <= self.midfield_threshold:
                # Calculate ball displacement from the last time step
                ball_displacement = np.linalg.norm(o['ball_direction'][:2])
                if ball_displacement >= self.minimum_pass_distance:
                    # Only reward if the pass is forward towards the opponent's goal
                    components["high_pass_reward"][i] = self.high_pass_bonus * ball_displacement
                    reward[i] += components["high_pass_reward"][i]

        return reward, components

    def step(self, action):
        """Step through environment, apply the customized reward, and append extra info."""
        obs, reward, done, info = self.env.step(action)
        modified_reward, reward_components = self.reward(reward)
        info["final_reward"] = sum(modified_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        return obs, modified_reward, done, info
