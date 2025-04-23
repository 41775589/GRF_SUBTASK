import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to augment the reward by encouraging agents to learn the technique of 
    stop-sprint and stop-moving. It monitors the use of sprint actions and abrupt stopping 
    when changing directions defensively.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and sticky actions counter.
        """
        observation = self.env.reset()
        self.sticky_actions_counter.fill(0)
        return observation

    def get_state(self, to_pickle):
        """
        Get the state for serialization.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        """
        Set the state for the environment with the serialization mechanism.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on the agent's performance in mastering stopping, sprinting, and movement.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if 'sticky_actions' in o:
                sprint_action = o['sticky_actions'][8]
                stop_action = (o['sticky_actions'][0] == 0 and o['sticky_actions'][4] == 0 and
                              o['active'] == 1)  # Assuming '1' is the identifier for when the player is actively moving

                # Reward for using sprint then stopping
                if sprint_action and stop_action:
                    components["stop_sprint_reward"][rew_index] = 0.5
                    reward[rew_index] += components["stop_sprint_reward"][rew_index]
                    
        return reward, components

    def step(self, action):
        """
        After performing an action, get the observations, modified rewards, determine the done status, and other info.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # To manage sticky actions, which are action behaviors that persist, such as dribbling or sprinting.
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                
        return observation, reward, done, info
