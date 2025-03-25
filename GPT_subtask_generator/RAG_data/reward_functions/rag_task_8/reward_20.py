import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for quick decision-making in initiating counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize reward parameters
        self.possession_regained = False
        self.initiate_counter_attack = False

    def reset(self):
        # Reset the internal states at the start of an episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_regained = False
        self.initiate_counter_attack = False
        return self.env.reset()

    def get_state(self, to_pickle):
        # Add the wrapper's specific state into the pickle
        to_pickle['possession_regained'] = self.possession_regained
        to_pickle['initiate_counter_attack'] = self.initiate_counter_attack
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Extract the state of the wrapper from the pickle
        from_picle = self.env.set_state(state)
        self.possession_regained = from_picle['possession_regained']
        self.initiate_counter_attack = from_picle['initiate_counter_attack']
        return from_picle

    def reward(self, reward):
        # Modify the reward according to the wrapper's rule
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' in o:
                if o['ball_owned_team'] == 0 and not self.possession_regained:
                    # Your team just regained possession; next is to check for quick counter-play initiation
                    self.possession_regained = True
                    components['possession_regained'] = 0.1
                    reward[rew_index] += components['possession_regained']
                elif o['ball_owned_team'] != 0:
                    self.possession_regained = False
                    self.initiate_counter_attack = False
                
                if self.possession_regained and not self.initiate_counter_attack:
                    if np.linalg.norm(o['ball_direction']) > 0.3:  # Arbitrary threshold to denote rapid movement
                        # Counter-attack likely initiated
                        self.initiate_counter_attack = True
                        components['initiate_counter_attack'] = 0.2  # Reward for quick counter-attack initiation
                        reward[rew_index] += components['initiate_counter_attack']

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
