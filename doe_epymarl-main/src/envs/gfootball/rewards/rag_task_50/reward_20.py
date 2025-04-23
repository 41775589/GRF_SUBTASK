import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for completing successful long passes
    between specified regions on the field, promoting vision, timing,
    and precision in ball distribution.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define regions on the field that act as source and target for long passes
        self.pass_regions = [
            {'source': [-1.0, -0.42, -0.2, 0], 'target': [0.2, 0.42, 1.0, -0.42]},  # left to right top
            {'source': [-1.0, 0, -0.2, 0.42], 'target': [0.2, -0.42, 1.0, 0]},  # left to right bottom
            {'source': [0.2, -0.42, 1.0, 0], 'target': [-1.0, 0, -0.2, 0.42]},  # right to left top
            {'source': [0.2, 0.42, 1.0, -0.42], 'target': [-1.0, -0.42, -0.2, 0]}  # right to left bottom
        ]
        self.pass_reward = 0.5
        self.last_ball_pos = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_pos = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.last_ball_pos
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_pos = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_pos = o['ball'][:2]  # just use x, y
            
            if self.last_ball_pos is not None and o['ball_owned_team'] == 1:
                dx, dy = current_ball_pos - self.last_ball_pos
                distance = np.sqrt(dx**2 + dy**2)
                
                # Check for a significant movement of the ball that indicates a long pass
                if distance > 0.5:
                    for region in self.pass_regions:
                        # Check if pass was initiated in a source region and received in a target region
                        if (self._in_region(self.last_ball_pos, region['source']) and
                            self._in_region(current_ball_pos, region['target'])):
                            reward[rew_index] += self.pass_reward
                            components["pass_reward"][rew_index] = self.pass_reward
    
            # Update the last ball position
            self.last_ball_pos = current_ball_pos
        
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info

    def _in_region(self, position, region):
        """
        Helper function to determine if a position (x,y) falls inside a defined rectangular region [x1, y1, x2, y2]
        """
        x, y = position
        x1, y1, x2, y2 = region
        return x1 <= x <= x2 and y1 <= y <= y2
