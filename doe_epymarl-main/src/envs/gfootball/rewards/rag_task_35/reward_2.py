import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic reward based on maintaining strategic positioning and transitioning between defensive and attacking strategies."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Adjust rewards based on strategic positioning and role switches
        for rew_index, o in enumerate(observation):
            # Reward for being in an optimal position for the current game mode
            if o['game_mode'] in (1, 3, 4, 6):  # These game modes involve set pieces or transitions
                positioning_score = self.evaluate_positioning(o, defensive=(o['game_mode'] in (3, 6)))
                components["positioning_reward"][rew_index] += positioning_score
                reward[rew_index] += components["positioning_reward"][rew_index]

            # Encourage switching between roles by checking transitions in game states or ball possession changes
            if o['game_mode'] == 0:  # Normal game play
                if self.sticky_actions_counter[8] > 0 and self.sticky_actions_counter[9] > 0:  # Sprint and dribble actions are indicators of role switches
                    transition_score = 0.1
                    reward[rew_index] += transition_score

        return reward, components

    def evaluate_positioning(self, observation, defensive):
        """Custom logic to evaluate player's positioning based on the game phase."""
        if defensive:
            # If defending, players should be closer to their own goal
            distance_to_goal = np.linalg.norm(observation['left_team'][observation['active']] + 1)  # -1 is the x-coordinate of the home goal
        else:
            # If attacking, players should advance towards the opponent's goal
            distance_to_goal = np.linalg.norm(observation['right_team'][observation['active']] - 1)  # 1 is the x-coordinate of the away goal
        return -distance_to_goal  # Negative because closer should be better, assuming normalize distances

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
