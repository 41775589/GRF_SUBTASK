import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances reward structure focused on offensive actions:
    - Encourages accurate shooting
    - Rewards dribbling skills that evade opponents
    - Promotes successful long and high passes
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Keep track of how often sticky actions are used

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save any important internal states
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore any important internal states
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Access the observations
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owner = o.get('ball_owned_player')

            # Check if the active player has the ball
            if o['ball_owned_team'] == 0 and ball_owner == o['active']:
                # Encourage shooting towards goal (y-coordinates near zero are close to the goal axis)
                if abs(o['ball'][1]) < 0.05 and o['ball'][0] > 0.5:
                    components['shooting_reward'][rew_index] = 1.0
                    reward[rew_index] += components['shooting_reward'][rew_index]

                # Reward dribbling: check if the action to dribble is employed by ball owner
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # index 9 refers to dribbling
                    components['dribbling_reward'][rew_index] = 0.1
                    reward[rew_index] += components['dribbling_reward'][rew_index]

                # Passing effectiveness: encouraging long and high passes if they switch the ball far forward
                if 'ball_direction' in o and o['ball_direction'][0] > 0.1 and abs(o['ball_direction'][1]) > 0.1:
                    components['passing_reward'][rew_index] = 0.2
                    reward[rew_index] += components['passing_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
