import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper specialized for enhancing the learning of agents focusing on performing effective
    standing tackles during normal play and set-piece scenarios without incurring penalties.
    The wrapper rewards precision in tackles and control over the ball recovery process.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.standing_tackles_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.standing_tackles_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['standing_tackles_counter'] = self.standing_tackles_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.standing_tackles_counter = from_pickle['standing_tackles_counter']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function to promote accurate and penalty-free tackling during gameplay.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for player_idx in range(len(reward)):
            o = observation[player_idx]
            
            # Tackle actions that lead to regaining control without fouls get rewarded.
            if o['game_mode'] in [0, 2, 4]:  # Normal, Free kick, Corner situations
                if o['ball_owned_team'] == 0 and (o['has_card'] == 0):  # No yellow or red cards
                    proximity_to_ball = np.linalg.norm(np.array(o['ball'][:2]) - np.array(o['left_team'][o['active']][:2]))
                    if proximity_to_ball < 0.1:  # Within close range to perform a tackle
                        components['tackle_precision_reward'][player_idx] = 1.0  # Static reward for precision
                        reward[player_idx] += components['tackle_precision_reward'][player_idx]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        return observation, reward, done, info
