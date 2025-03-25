import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces rewards for effectively managing player stamina through efficient sprint and movement actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_actions_used = 0
        self.sprint_actions_used = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_actions_used = 0
        self.sprint_actions_used = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StopSprintMoveStates'] = (self.stop_actions_used, self.sprint_actions_used)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.stop_actions_used, self.sprint_actions_used = from_pickle['StopSprintMoveStates']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "conservation_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward efficient use of stop and sprint actions
            if 'sticky_actions' in o:
                sprint = o['sticky_actions'][8]   # Sprint action index
                dribble = o['sticky_actions'][9]  # Dribble action index

                if sprint == 1:
                    self.sprint_actions_used += 1
                else:
                    # Reward if player is managing stamina by not sprinting unnecessarily
                    components['conservation_reward'][rew_index] += 1.0

                if dribble == 0:
                    self.stop_actions_used += 1
                else:
                    # Penalize for constant dribbling which might tire the player
                    components['conservation_reward'][rew_index] -= 0.5

                reward[rew_index] += components['conservation_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
