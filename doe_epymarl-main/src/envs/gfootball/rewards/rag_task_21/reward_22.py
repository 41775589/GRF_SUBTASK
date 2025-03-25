import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward specifically for defensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.ball_intercepted = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.ball_intercepted = False
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position,
            'ball_intercepted': self.ball_intercepted
        }
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        self.ball_intercepted = from_pickle['CheckpointRewardWrapper']['ball_intercepted']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'interception_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            current_ball_position = np.array(obs['ball'])
            
            # Check if the ball position has changed substantially which can indicate a pass or shot
            if self.last_ball_position is not None:
                ball_moved = np.linalg.norm(current_ball_position - self.last_ball_position) > 0.1

                # Check for interception when ball ownership changes to defending team
                if (ball_moved and obs['ball_owned_team'] in [0, 1] and
                    self.last_ball_owned_team in [0, 1] and
                    self.last_ball_owned_team != obs['ball_owned_team']):
                    
                    components['interception_reward'][i] = 1.0
                    reward[i] += components['interception_reward'][i]
                    self.ball_intercepted = True

            self.last_ball_position = current_ball_position
            self.last_ball_owned_team = obs['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
