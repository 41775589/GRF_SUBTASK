import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds reward for mastering stop-and-sprint techniques in defensive movements."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_actions = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_actions = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_actions': self.previous_actions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_actions = from_pickle['CheckpointRewardWrapper']['previous_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward for stop-and-sprint strategy
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            action_sprint = o['sticky_actions'][8]  # Sprint action
            action_dribble = o['sticky_actions'][9] # Dribble action

            # Reward for stopping sprint abruptly and then moving
            if self.previous_actions is not None:
                previous_sprint = self.previous_actions[rew_index]['sticky_actions'][8]
                
                # If was sprinting and now stopped, and either moving directionally or dribbling
                if previous_sprint and not action_sprint and (action_dribble or np.any(o['sticky_actions'][:8])):
                    components["stop_sprint_reward"][rew_index] = 0.1
                    reward[rew_index] += components["stop_sprint_reward"][rew_index]

        self.previous_actions = observation
        
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
