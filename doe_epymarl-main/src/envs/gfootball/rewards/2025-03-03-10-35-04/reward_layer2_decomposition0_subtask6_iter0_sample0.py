import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom reward wrapper focused on mastering Short Pass techniques under game pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.short_pass_completion = 0.3  # rewarding completing a short pass
        self.ball_position_score_factor = 0.1  # how closely position matches the target

    def reset(self):
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

        # Each agent's metrics extracted from the observation for the football environment
        o = observation[0]  # Assuming our agent index is 0 as we have a single-agent focus here
        components['short_pass_reward'] = [0.0]
        
        # reward successful short passes especially under game pressure (game_mode != 0)
        if o['game_mode'] == 0 and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
            components['short_pass_reward'][0] = self.short_pass_completion
            
            # Check target position is close to another team player 
            target_position = o['right_team'][o['designated']][:2]  # Assumes target is within own team
            ball_position = o['ball'][:2]

            # euclidean distance to measure closeness
            distance_to_target = np.sqrt((target_position[0] - ball_position[0])**2 + \
                                         (target_position[1] - ball_position[1])**2)
            components['ball_position_score'] = [max(0, self.ball_position_score_factor * (1 - distance_to_target))]

        # Adding all components to get final reward
        final_reward = sum([components[key][0] for key in components.keys()])
        reward[0] = final_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        return observation, reward, done, info
