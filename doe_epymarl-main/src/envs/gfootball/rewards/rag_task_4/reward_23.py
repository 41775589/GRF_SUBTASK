import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for advanced dribbling and sprinting skills.
    The reward is designed to motivate the agent to dribble the ball through tight defensive lines
    with speed and control, utilizing sprints effectively.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize dribbling and sprinting counters
        self.dribbling_sprint_periods = {}

    def reset(self):
        """
        Reset the reward wrapper state for a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_sprint_periods = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the wrapper state with the environment state. Used for serialization.
        """
        to_pickle['CheckpointRewardWrapper_states'] = self.dribbling_sprint_periods
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load the wrapper state along with the environment state. Used for deserialization.
        """
        from_pickle = self.env.set_state(state)
        self.dribbling_sprint_periods = from_pickle['CheckpointRewardWrapper_states']
        return from_pickle

    def reward(self, reward):
        """
        Computes the enhanced reward using the observation data from the environment.
        This promotes dribbling combined with sprinting as a skill.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy()}

        if observation is None:
            return reward, components
        
        new_rewards = []
        components['advanced_dribbling_and_sprinting_reward'] = []
        
        for agent_obs in observation:
            base_reward = reward
            
            # Check if player is dribbling and sprinting.
            dribbling = agent_obs['sticky_actions'][9] > 0  # action_dribble index
            sprinting = agent_obs['sticky_actions'][8] > 0  # action_sprint index
            
            dribble_sprint_reward = 0.0
            if dribbling and sprinting:
                dribble_sprint_reward = 0.1
                step_id = agent_obs.get('steps_left', 3000)
                period_key = step_id // 300  # discretize time for simplification
                if not self.dribbling_sprint_periods.get(period_key, False):
                    dribble_sprint_reward += 0.2  # additional reward for new period
                    self.dribbling_sprint_periods[period_key] = True
            
            new_reward = base_reward + dribble_sprint_reward
            new_rewards.append(new_reward)
            components['advanced_dribbling_and_sprinting_reward'].append(dribble_sprint_reward)
        
        return new_rewards, components

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
