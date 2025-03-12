import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive skills like shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 0.3
        self.dribbling_reward = 0.2
        self.passing_reward = 0.1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'shooting_reward': self.shooting_reward,
                                                'dribbling_reward': self.dribbling_reward,
                                                'passing_reward': self.passing_reward}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_reward = from_pickle['shooting_reward']
        self.dribbling_reward = from_pickle['dribbling_reward']
        self.passing_reward = from_pickle['passing_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'offensive_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index in range(len(reward)):
            o = observation[index]
            active_player = o['active']
            if o['ball_owned_player'] == active_player:
                # Dribbling reward for moving with the ball
                if any(o['sticky_actions'][8:10]):  # Check sprint and dribble actions
                    components['offensive_reward'][index] += self.dribbling_reward
                    reward[index] += components['offensive_reward'][index]

                # Simulate shooting reward for attempts close to the goal
                goal_distance = np.abs(o['ball'][0] - 1)  # Assuming shooting towards right goal at x=1
                if goal_distance < 0.1 and o['ball'][2] > 0.1:  # Ball in the air in front of the goal
                    components['offensive_reward'][index] += self.shooting_reward
                    reward[index] += components['offensive_reward'][index]

                # Passing reward for changes in ball ownership with low ball distance
                if o['sticky_actions'][6]:
                    components['offensive_reward'][index] += self.passing_reward
                    reward[index] += components['offensive_reward'][index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
