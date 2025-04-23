import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing accurate long passes in different areas of the field."""

    def __init__(self, env):
        super().__init__(env)
        self.last_ball_pos = None
        self.pass_accuracy_reward = 1.0
        self.long_pass_threshold = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.last_ball_pos = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_pos'] = self.last_ball_pos
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_pos = from_pickle.get('last_ball_pos', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward}

        components = {'base_score_reward': reward.copy(),
                      'pass_accuracy_reward': [0.0, 0.0]}

        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'ball_owned_player' in o:
                active_player_pos = o['left_team'][o['active']]
                
                # Check if this is a new possession or continue of old
                if self.last_ball_pos is not None:
                    # Calculate distance ball travelled
                    ball_distance = np.linalg.norm(o['ball'][:2] - self.last_ball_pos)
                    
                    # Check if it is a long pass and if possession has changed
                    if ball_distance >= self.long_pass_threshold:
                        components['pass_accuracy_reward'][i] += self.pass_accuracy_reward
                        reward[i] += self.pass_accuracy_reward
                
                # Update last ball position
                self.last_ball_pos = o['ball'][:2]
            else:
                # No possession, forget last ball position
                self.last_ball_pos = None

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        final_reward_total = sum(reward)
        
        # Adding final reward and component rewards to info
        info.update({
            "final_reward": final_reward_total,
            **{f"component_{k}": sum(v) for k, v in components.items()}
        })
        
        return observation, reward, done, info
