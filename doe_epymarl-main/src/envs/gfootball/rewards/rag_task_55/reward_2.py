import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically for enhancing defensive tactics via tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize the counter for sticky actions 
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Parameters for defensive rewards
        self.standing_tackle_reward = 0.5
        self.sliding_tackle_reward = 1.0
        self.foul_penalty = -0.5
        self.tackle_efficiency_threshold = 3  # Threshold of tackles before fouling is likely

        # State counters for rewards
        self.tackle_count = 0
        self.foul_count = 0

    def reset(self):
        self.tackle_count = 0
        self.foul_count = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackle_count'] = self.tackle_count
        to_pickle['foul_count'] = self.foul_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_count = from_pickle['tackle_count']
        self.foul_count = from_pickle['foul_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Get current observations

        # Initialize components of rewards and continue if no observation
        components = {"base_score_reward": reward.copy(),
                      "standing_tackle_reward": 0.0,
                      "sliding_tackle_reward": 0.0,
                      "foul_penalty": 0.0}
        if observation is None:
            return reward, components

        # Loop through each agent (note: assuming 2 agents here for simplicity)
        for agent_index, obs in enumerate(observation):
            if 'game_mode' in obs and obs['game_mode'] == 3:  # Free Kick against the agent
                # Assume foul has occurred if game mode switches to Free Kick
                if self.tackle_count >= self.tackle_efficiency_threshold:
                    reward[agent_index] += self.foul_penalty
                    components["foul_penalty"] = self.foul_penalty
                    self.foul_count += 1
                self.tackle_count = 0

            # Tackle is performed - checking sticky actions: 10 is stand_tackle, 11 is sliding_tackle (assumed indices)
            if obs['sticky_actions'][10]:
                reward[agent_index] += self.standing_tackle_reward
                components["standing_tackle_reward"] = self.standing_tackle_reward
                self.tackle_count += 1

            if obs['sticky_actions'][11]:
                reward[agent_index] += self.sliding_tackle_reward
                components["sliding_tackle_reward"] = self.sliding_tackle_reward
                self.tackle_count += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
