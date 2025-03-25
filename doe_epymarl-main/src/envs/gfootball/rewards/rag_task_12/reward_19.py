import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This Reward Wrapper focuses on training an agent that excels as a midfielder/advanced defender.
    It emphasizes effective ball handling skills like High Pass, Long Pass, Dribble, and dynamic
    Sprinting abilities. 
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.3
        self.dribble_reward = 0.2
        self.sprint_reward = 0.15

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of this reward wrapper along with environment's state.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of this reward wrapper from the state retrieved from the environment.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward logic to enhance the agent's midfield/defensive capabilities.
        Rewards are associated with successful passes, dribbling, and strategic sprint management.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward).copy(),
            "pass_reward": np.zeros_like(reward),
            "dribble_reward": np.zeros_like(reward),
            "sprint_reward": np.zeros_like(reward)
        }

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            if 'sticky_actions' in o:
                # High Pass and Long Pass are scaled heavily.
                if o['sticky_actions'][6]:
                    components["pass_reward"][idx] = self.pass_reward
                if o['sticky_actions'][7]:
                    components["pass_reward"][idx] += self.pass_reward
                
                # Dribble under control to manage game dynamic.
                if o['sticky_actions'][9]:
                    components["dribble_reward"][idx] = self.dribble_reward

                # Effective sprint management for transitions.
                if o['sticky_actions'][8]:
                    components["sprint_reward"][idx] = self.sprint_reward

                # Update the reward for this step.
                reward[idx] += components["pass_reward"][idx] + components["dribble_reward"][idx] + components["sprint_reward"][idx]

        return reward, components

    def step(self, action):
        """
        Step the environment and modify reward with additional rewards for checking transitions,
        passes, dribbles, and sprints.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
