import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for dribbling and sprinting actions to promote
    offensive penetration and ball control techniques.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_reward = 0.05
        self.sprint_reward = 0.05

    def reset(self):
        """
        Resets the environment and sticky action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment for persistence.
        """
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment from persisted values.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """
        Enhances the incoming reward by adding bonuses for using dribble and sprint actions
        effectively during tight control scenarios or offensive movements.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        components['dribbling_reward'] = [0.0] * len(reward)
        components['sprint_reward'] = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_control = o['ball_owned_player'] == o['active']
            if ball_control and o['sticky_actions'][9]:  # Index 9 is 'action_dribble'
                components['dribbling_reward'][rew_index] += self.dribbling_reward
            if ball_control and o['sticky_actions'][8]:  # Index 8 is 'action_sprint'
                components['sprint_reward'][rew_index] += self.sprint_reward

            reward[rew_index] += (
                components['dribbling_reward'][rew_index] +
                components['sprint_reward'][rew_index]
            )

        return reward, components

    def step(self, action):
        """
        Takes a step in the environment applying the action and recalculating the reward.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Include component values in 'info' for introspection
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            if isinstance(agent_obs, dict):
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
                    info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
