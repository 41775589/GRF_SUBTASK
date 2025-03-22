import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a customized reward focused on offensive strategies: shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.shoot_reward = 1.0
        self.dribble_reward = 0.5
        self.pass_reward = 0.3
        self.goal_score_reward = 5.0

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_rewards_state'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any internal state if necessary
        return from_pickle

    def reward(self, reward):
        components = {
            'base_score_reward': reward.copy(),
            'shoot_reward': [0.0] * len(reward),
            'dribble_reward': [0.0] * len(reward),
            'pass_reward': [0.0] * len(reward),
            'goal_score_reward': [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'sticky_actions' in o:
                sticky_actions = o['sticky_actions']

                # Rewarding shooting action - typically last action in sticky_actions array
                if sticky_actions[-1] == 1:  # Assuming shooting is the last sticky action
                    components['shoot_reward'][rew_index] = self.shoot_reward
                    reward[rew_index] += components['shoot_reward'][rew_index]

                # Reward dribbling by checking dribble action index
                if sticky_actions[9] == 1:  # Assuming dribbling index is 9
                    components['dribble_reward'][rew_index] = self.dribble_reward
                    reward[rew_index] += components['dribble_reward'][rew_index]

                # Reward passing using a hypothetical index for passing action
                if sticky_actions[6] == 1:  # Assuming passing index is 6
                    components['pass_reward'][rew_index] = self.pass_reward
                    reward[rew_index] += components['pass_reward'][rew_index]

            # Add additional reward for scoring a goal
            score = o['score']
            if score[1] > score[0]: # Assuming index 1 is the scoring team
                components['goal_score_reward'][rew_index] = self.goal_score_reward
                reward[rew_index] += components['goal_score_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
