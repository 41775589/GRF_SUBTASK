import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for wide midfield responsibilities including mastering
    high pass and positioning to expand the field of play.
    """
    def __init__(self, env):
        super().__init__(env)
        # Keeping track of player's previous and current positions and actions
        self.prev_positions = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward components multipliers
        self.wide_play_multiplier = 0.1
        self.high_pass_multiplier = 0.2
        self.positioning_multiplier = 0.05

    def reset(self):
        """
        Reset the wrapper state and the environment
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Include the wrapper state to the environment state for serialization.
        """
        to_pickle['CheckpointRewardWrapper'] = {
            'prev_positions': self.prev_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state including wrapper specific states.
        """
        from_pickle = self.env.set_state(state)
        self.prev_positions = from_pickle['CheckpointRewardWrapper'].get('prev_positions', {})
        return from_pickle

    def reward(self, reward):
        """
        Implement a custom reward function focusing on wide midfield roles.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "wide_play_reward": [0.0] * len(reward),
            "high_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['right_team'][o['active']]
            player_role = o['right_team_roles'][o['active']]

            # Reward for wide midfielders
            if player_role in [6, 7]:  # Assuming 6 and 7 are lateral midfield positions
                # Positioning - Encourage spreading out horizontally near the sidelines
                if abs(player_pos[1]) > 0.3:  # closer to the sidelines
                    components["positioning_reward"][rew_index] = self.positioning_multiplier
                    reward[rew_index] += components["positioning_reward"][rew_index]

                # Check if high pass was made
                if 'high_pass' in o['sticky_actions'] and o['sticky_actions'][9]:
                    components["high_pass_reward"][rew_index] = self.high_pass_multiplier
                    reward[rew_index] += components["high_pass_reward"][rew_index]

                # Wide play encouragement - staying wide when team has possession
                if o['ball_owned_team'] == 1:  # Assuming the right team is the agent's team
                    if abs(player_pos[1]) > 0.3:
                        components["wide_play_reward"][rew_index] = self.wide_play_multiplier
                        reward[rew_index] += components["wide_play_reward"][rew_index]

                # Save the current position for the next step comparison
                self.prev_positions[rew_index] = player_pos

        return reward, components

    def step(self, action):
        """
        Take a step in the environment and add custom reward logic
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
