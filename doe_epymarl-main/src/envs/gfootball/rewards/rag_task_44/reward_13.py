import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful stop-dribble execution and defensive position maintenance."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_dribble_executed = False
        self.defensive_action_performed = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_dribble_executed = False
        self.defensive_action_performed = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['RewardWrapper'] = {
            'stop_dribble_executed': self.stop_dribble_executed,
            'defensive_action_performed': self.defensive_action_performed
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.stop_dribble_executed = from_pickle['RewardWrapper']['stop_dribble_executed']
        self.defensive_action_performed = from_pickle['RewardWrapper']['defensive_action_performed']
        return from_pickle

    def reward(self, reward):
        """Customize reward: additional points for correctly executed stop-dribble tactics under pressure."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0] * len(reward),
            "defensive_position_reward": [0.0] * len(reward)
        }

        # ensure abort if no observation is available
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for stopping dribble
            if 'sticky_actions' in o:
                is_dribbling = o['sticky_actions'][9]
                if is_dribbling and not self.stop_dribble_executed:
                    components["stop_dribble_reward"][rew_index] = 0.5
                    reward[rew_index] += components["stop_dribble_reward"][rew_index]
                    self.stop_dribble_executed = True

            # Reward for maintaining defensive position
            if 'left_team' in o and o['active'] in o['left_team']:
                is_defensive = all(abs(o['left_team'][o['active']][1]) < 0.1)  # y-coord close to goal
                if is_defensive and not self.defensive_action_performed:
                    components["defensive_position_reward"][rew_index] = 0.3
                    reward[rew_index] += components["defensive_position_reward"][rew_index]
                    self.defensive_action_performed = True

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
