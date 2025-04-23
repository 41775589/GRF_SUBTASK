import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically designed for training agents specialized 
       in perfecting standing tackles and enhancing possession regaining during normal gameplay 
       and set-piece defense scenarios without risking penalties."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_rewards = 0.0
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Iterate over observations for all agents (in our case it's usually 2 players)
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            has_ball = (o['ball_owned_team'] == 0)
            opponent_has_ball = (o['ball_owned_team'] == 1)
            active_player = o['active']
            game_mode_normal = (o['game_mode'] == 0)  # Normal gameplay
            
            # Reward standing tackles when player doesn't commit a foul (assumed by game mode)
            if game_mode_normal:
                if opponent_has_ball and 'ball_owned_player' in o and active_player == o['ball_owned_player']:
                    success_tackle = 'action_tackle' in o['sticky_actions'] and o['sticky_actions']['action_tackle']
                    if success_tackle:
                        components["tackle_reward"] = 0.5
                        reward[rew_index] += components["tackle_reward"]

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
