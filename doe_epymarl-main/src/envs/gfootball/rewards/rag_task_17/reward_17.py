import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on the specific tasks of wide midfield roles:
    Emphasizes mastery over High Pass actions, maintaining correct position to expand the play 
    and aiding in lateral transitions to stretch the opponent’s defense.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize the action counters and positioning rewards
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_effectiveness = 0.5

    def reset(self):
        # Reset action counters on environment reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save state if needed for logging or restoring later
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'high_pass_effectiveness': self.high_pass_effectiveness
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore state from saved data
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.high_pass_effectiveness = from_pickle['CheckpointRewardWrapper']['high_pass_effectiveness']
        return from_pickle

    def reward(self, reward):
        # Start with initial environment generated reward and adapt based on metrics
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # High pass reward heuristic - the action should be high_pass and have a high influence
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Checking if the high_pass action is on
                components["high_pass_reward"][rew_index] += self.high_pass_effectiveness

            # Positioning reward - being on the lateral sides and showing activity
            if 'right_team' in o:
                player_x, player_y = o['right_team'][o['active']]
                # Encourage reaching the sides of the pitch for wide midfielders
                if abs(player_y) >= 0.3:  
                    components["positioning_reward"][rew_index] += 0.3

            # Aggregate the rewards
            total_reward = (components["base_score_reward"][rew_index] +
                            components["high_pass_reward"][rew_index] +
                            components["positioning_reward"][rew_index])

            reward[rew_index] = total_reward
        
        return reward, components

    def step(self, action):
        # Perform a step in the environment, retrieve observations and modify the reward
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
