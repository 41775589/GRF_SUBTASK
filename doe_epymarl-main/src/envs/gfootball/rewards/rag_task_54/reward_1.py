import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance the effectiveness of collaborative plays between shooters and passers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_bonus = 0.1
        self.shot_bonus = 0.2
        self.pass_to_shot_bonus = 0.5
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Returns the state along with internal state variable."""
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Sets the state internal variables."""
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle.get('previous_ball_owner', None)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": [0.0] * len(reward),
                      "shot_bonus": [0.0] * len(reward),
                      "pass_to_shot_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        ball_owner_team = observation['ball_owned_team']
        if ball_owner_team not in [0, 1]:
            return reward, components
        
        current_ball_owner = observation['ball_owned_player']
        if 'ball_owned_player' in observation and current_ball_owner != -1:
            if self.previous_ball_owner is not None and self.previous_ball_owner != current_ball_owner:
                # A pass has occurred
                components['pass_bonus'][ball_owner_team] = self.pass_bonus
                reward[ball_owner_team] += self.pass_bonus
                
            # Check if a shot on goal is made
            if 'action' in observation and observation['action'] == 12: # 12 being the index for 'shot'
                components['shot_bonus'][ball_owner_team] += self.shot_bonus
                reward[ball_owner_team] += self.shot_bonus
                if self.previous_ball_owner is not None:
                    # This includes an indirect bonus for a pass followed by a shot
                    components['pass_to_shot_bonus'][ball_owner_team] += self.pass_to_shot_bonus
                    reward[ball_owner_team] += self.pass_to_shot_bonus
        
        self.previous_ball_owner = current_ball_owner
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
