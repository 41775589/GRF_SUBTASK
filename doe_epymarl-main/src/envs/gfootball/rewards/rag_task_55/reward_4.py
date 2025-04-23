import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive tactics through successful tackles."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sliding_tackle_reward = 0.2
        self.standing_tackle_reward = 0.1
        self.non_foul_play_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Verify rewards for both agents
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Identify the game mode to adjust rewards for tackling without foul
            if o['game_mode'] in {2, 3, 5, 6}:  # Tackle relevant modes: FreeKick, Corner, ThrowIn, Penalty
                if o['ball_owned_team'] == -1 or (o['ball_owned_team'] == 1 and o['sticky_actions'][0] == 1): # standing tackle
                    components["tackle_reward"][rew_index] = self.standing_tackle_reward
                if o['ball_owned_team'] == 1 and o['sticky_actions'][1] == 1: # sliding tackle
                    components["tackle_reward"][rew_index] = self.sliding_tackle_reward
                
                # Reward for non-foul outcomes post-tackle
                if o['game_mode'] != o['prev_game_mode'] and o['prev_game_mode'] == 2: 
                    reward[rew_index] += self.non_foul_play_reward

            reward[rew_index] += self.sticky_actions_counter[rew_index] * (
                components["tackle_reward"][rew_index])

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
