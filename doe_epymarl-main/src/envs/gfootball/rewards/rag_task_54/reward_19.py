import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that encourages collaborative play by rewarding passes 
    that set up scoring opportunities and successful shots on goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_to_shot_multiplier = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "passing_score": [0.0] * len(reward),
            "shooting_score": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            if obs['game_mode'] in {1, 6}:  # Evaluate during Normal and Penalty Kick play
                # Check for a successful pass leading to a potential shot
                if obs['ball_owned_team'] == 1 and obs['ball_owned_player'] is not None:
                    player_x, player_y = obs['right_team'][obs['ball_owned_player']]
                    
                    if abs(player_x) > 0.8:  # Closer to opponent's goal
                        components['passing_score'][idx] = self.pass_to_shot_multiplier
                # Check for successful shots based on proximity to opponent's goal
                ball_x, ball_y = obs['ball'][0], obs['ball'][1]
                if abs(ball_x) > 0.9:  # Very close to a goal line
                    components['shooting_score'][idx] += 1.0  # basic score for being in a critical area
                
            reward[idx] += components['passing_score'][idx] + components['shooting_score'][idx]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Update info dict with reward components and final aggregated reward
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
