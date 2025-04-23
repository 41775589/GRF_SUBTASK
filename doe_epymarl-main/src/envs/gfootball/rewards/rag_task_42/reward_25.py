import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering midfield dynamics including enhanced coordination under pressure and strategic repositioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_transition_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reset the dictionary storing the midfield transition rewards for each episode
        self.midfield_transition_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_transition_rewards'] = self.midfield_transition_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_transition_rewards = from_pickle.get('midfield_transition_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos_x = o['ball'][0]
            # Define midfield region as the center third of the pitch
            midfield_start, midfield_end = -1/3, 1/3
            is_ball_in_midfield = midfield_start <= ball_pos_x <= midfield_end

            # Check for strategic transitions in midfield
            if is_ball_in_midfield and not self.midfield_transition_rewards.get(rew_index, False):
                strategic_transition = False
                
                # Check if the ball was previously in an outside third and is now in the middle third
                # This is a proxy for good coordination under pressure and strategic repositioning
                previous_ball_pos_x = self.sticky_actions_counter[rew_index]  # Assume this stores the ball's last x position
                if previous_ball_pos_x < midfield_start or previous_ball_pos_x > midfield_end:
                    strategic_transition = True

                if strategic_transition:
                    components["midfield_transition_reward"][rew_index] = 0.5  # Reward for successful midfield transition
                    reward[rew_index] += components["midfield_transition_reward"][rew_index]
                    self.midfield_transition_rewards[rew_index] = True

            # Update the previous position to current position
            self.sticky_actions_counter[rew_index] = ball_pos_x

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add all components to 'info' for logging if needed
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
