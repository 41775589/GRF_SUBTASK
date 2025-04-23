import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for training a goalkeeper in various tasks such as shot stopping, decision making for ball distribution, and effective communication with defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prevent_goals = 0.5
        self.distribute_ball = 0.3
        self.communicate_defense = 0.2

    def reset(self):
        """Resets the environment and the internal metrics."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment for deserialization."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Rewards the agent for shot stopping, ball distribution decisions, and defense communication."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "prevent_goals": [0.0],
            "distribute_ball": [0.0],
            "communicate_defense": [0.0]
        }
        
        if observation is None:
            return reward, components

        player_role = observation['right_team_roles'][observation['active']]
        if player_role == 0:  # Goalkeeper
            goalie_action = observation['sticky_actions']
            if 'ball_owned_team' in observation and observation['ball_owned_team'] == 1:
                # prevent goals
                if observation['ball'][0] > 0.9:  # near the goal area on x-axis
                    components["prevent_goals"][0] = self.prevent_goals
        
                # distribute ball
                if goalie_action[4] or goalie_action[5]:  # right or left pass
                    components["distribute_ball"][0] = self.distribute_ball
        
            # communicate defense
            # Suppose communication is triggered by non-standard actions like sliding or high pass
            if goalie_action[6] or goalie_action[7]:  # slide or high pass
                components["communicate_defense"][0] = self.communicate_defense
        
        for key in components:
            reward += components[key][0]
        
        return reward, components

    def step(self, action):
        """Take a step using the wrapped environment and modify the outputs with the new reward function."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
