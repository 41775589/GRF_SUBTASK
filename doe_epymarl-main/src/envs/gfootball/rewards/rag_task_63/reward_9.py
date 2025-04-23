import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward signal for training a goalkeeper."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stopped = False
        self.has_controlled_ball = False
        self.communicated_with_defenders = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stopped = False
        self.has_controlled_ball = False
        self.communicated_with_defenders = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shot_stopped'] = self.shot_stopped
        to_pickle['has_controlled_ball'] = self.has_controlled_ball
        to_pickle['communicated_with_defenders'] = self.communicated_with_defenders
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_stopped = from_pickle.get('shot_stopped', False)
        self.has_controlled_ball = from_pickle.get('has_controlled_ball', False)
        self.communicated_with_defenders = from_pickle.get('communicated_with_defenders', False)
        return from_pickle  

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "shot_stopping_reward": [0.0] * len(reward),
            "ball_control_reward": [0.0] * len(reward),
            "communication_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        rewards = reward.copy()

        for rew_index in range(len(rewards)):
            o = observation[rew_index]
            goalie_index = np.argmin(o['left_team'][:, 0])  # assuming the goalie is the leftmost player
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == goalie_index:
                if not self.has_controlled_ball:
                    components['ball_control_reward'][rew_index] = 1.0
                    self.has_controlled_ball = True
            
            if o['game_mode'] == 6 and rew_index == goalie_index:  # Penalty mode
                if o['ball_owned_team'] == -1:  # Ball not owned after penalty kick
                    components['shot_stopping_reward'][rew_index] = 1.0
                    self.shot_stopped = True
            
            if o['designated'] == goalie_index: # Assuming designation as a form of communication
                if not self.communicated_with_defenders:
                    components['communication_reward'][rew_index] = 0.5
                    self.communicated_with_defenders = True

            rewards[rew_index] += sum(components[k][rew_index] for k in components)

        return rewards, components

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
