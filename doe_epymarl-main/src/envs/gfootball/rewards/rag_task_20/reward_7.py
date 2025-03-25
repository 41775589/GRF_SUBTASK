import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward signals to foster offensive strategies and team coordination."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.1
        self.shooting_bonus = 0.3
        self.positioning_bonus = 0.05

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
        # Accessing the latest observations
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(), 
                      "passing_bonus": [0.0] * len(reward),
                      "shooting_bonus": [0.0] * len(reward),
                      "positioning_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            if obs['game_mode'] == 1:  # Check if it's a KickOff
                # Encouraging passing at KickOff to set gameplay
                if obs['sticky_actions'][-1]: # action_dribble is active
                    components['passing_bonus'][rew_index] = self.passing_bonus
            
            # Check for shooting towards goal or effective positioning
            ball_owned_team = obs['ball_owned_team']
            team = 'left_team' if ball_owned_team == 0 else 'right_team'

            if ball_owned_team == obs['active'] and obs['ball'][0] > 0.5:
                # Ball is in the opponent's half, add positioning bonus
                components['positioning_bonus'][rew_index] = self.positioning_bonus
                
                if obs['game_mode'] == 0 and obs['sticky_actions'][6]:  # action_bottom (akin to shooting)
                    # If close enough for a goal shot
                    if np.linalg.norm(obs['ball'][1] - obs['score'][1]) < 0.1:
                        components['shooting_bonus'][rew_index] = self.shooting_bonus

            # Accumulate the reward components
            reward[rew_index] += (components['passing_bonus'][rew_index] +
                                  components['shooting_bonus'][rew_index] +
                                  components['positioning_bonus'][rew_index])
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
