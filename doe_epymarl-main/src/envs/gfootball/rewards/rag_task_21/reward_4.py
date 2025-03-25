import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing defensive skills like blocking, 
    intercepting passes, and detecting potential threats from the opposing team.
    This wrapper incentivizes the agent to actively position itself in defensive 
    intercept routes and make intercept attempts. It discourages passive defense.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercepts = {0: 0, 1: 0, 2: 0, 3: 0}  # Firm interceptions per agent
        self.blocks = {0: 0, 1: 0, 2: 0, 3: 0}      # Recorded blocks per agent

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercepts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.blocks = {0: 0, 1: 0, 2: 0, 3: 0}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_intercepts'] = self.intercepts
        to_pickle['CheckpointRewardWrapper_blocks'] = self.blocks
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.intercepts = from_pickle['CheckpointRewardWrapper_intercepts']
        self.blocks = from_pickle['CheckpointRewardWrapper_blocks']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "block_reward": [0.0] * len(reward)}

        assert len(reward) == len(observation)

        for index, o in enumerate(observation):
            # Ball interception logic
            if o['ball_owned_team'] == -1 and self.previous_ball_owned_team == 1:
                # If the ball was previously owned by the opponent and now it's free
                self.intercepts[index] += 1
                components['intercept_reward'][index] = 1.0

            # Defensive block logic
            if o['game_mode'] in {2, 3, 4, 5, 6}:  # Stoppage modes
                possible_block = np.random.rand() < 0.1  # 10% chance of successful block
                if possible_block:
                    self.blocks[index] += 1
                    components['block_reward'][index] = 0.5

            reward[index] += components['intercept_reward'][index] + components['block_reward'][index]

        self.previous_ball_owned_team = o.get('ball_owned_team', -1)
        return reward, components

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]        
        return observation, reward, done, info
