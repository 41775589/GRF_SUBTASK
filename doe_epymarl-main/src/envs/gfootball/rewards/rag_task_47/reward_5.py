import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering sliding tackles during counter-attacks and high-pressure situations."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking the frequency of each sticky action

    def reset(self):
        """ Reset the sticky action counters when the environment is reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Include sticky action counters in the state pickle. """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restore the sticky action counters from the pickle. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """ Modify the reward to emphasize mastering sliding tackles during counter-attacks in defensive third. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}  # Keep a copy of the base score reward
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Important metrics for this task
            player_x_pos = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]
            ball_x_pos = o['ball'][0]
            sticky_action_tackle = o['sticky_actions'][0]  # assuming index 0 is the tackle action

            # Enhance the reward when performing a tackle in the defensive third during counter-attacks
            if sticky_action_tackle == 1 and player_x_pos > 0.5 and abs(ball_x_pos - player_x_pos) < 0.1:
                components["tackle_reward"] = 1.0  # Reward for a properly timed tackle near our defensive third
            else:
                components["tackle_reward"] = 0.0

            # Update the total reward with the extra component for tackling
            reward[rew_index] += components["tackle_reward"] * 0.5  # Scaling factor for the tackle reward

        return reward, components

    def step(self, action):
        """Collect rewards, augment with component information, and return results after taking an action."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
