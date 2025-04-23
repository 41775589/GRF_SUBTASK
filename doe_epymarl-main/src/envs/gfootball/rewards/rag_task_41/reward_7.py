import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on encouraging efficient attacking strategies and creative plays.
       It provides a dense reward signal by promoting position handling, offensive maneuvers,
       and creativity under pressure from defensive setups."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to calibrate the importance of creativity and avoiding defenders
        self.creativity_reward = 0.2
        self.avoid_defender_reward = 0.1
        self.long_shot_reward = 0.15

    def reset(self):
        """Resets the environment to start a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the current state of the environment with the wrapper specific data."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the environment including wrapper specific data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Customize the reward function to encourage creative and effective offensive play."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "creativity_reward": [],
            "avoid_defender_reward": [],
            "long_shot_reward": []
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            base_reward = reward[i]
            
            # Reward players for performing creative plays (like risky passes/dribbles towards goal)
            creativity_bonus = self.creativity_reward if o['sticky_actions'][9] == 1 else 0  # Assuming dribble is a creative action
            
            # Reward for player proximity to goal while avoiding defenders
            dist_to_goal = np.abs(o['ball'][0] - 1)  # Assuming x=1 is the goal for the player's team
            defenders_nearby = any(np.linalg.norm(np.array(defender) - np.array(o['ball'][:2])) < 0.1 for defender in o['right_team'])  # Example threshold
            avoid_defenders_bonus = self.avoid_defender_reward if dist_to_goal < 0.3 and not defenders_nearby else 0
            
            # Additional reward for successful long shots
            long_shot_bonus = self.long_shot_reward if dist_to_goal > 0.5 and base_reward > 0 else 0
            
            # Calculate final reward for this agent
            total_reward = base_reward + creativity_bonus + avoid_defenders_bonus + long_shot_bonus
            
            reward[i] = total_reward
            components["creativity_reward"].append(creativity_bonus)
            components["avoid_defender_reward"].append(avoid_defenders_bonus)
            components["long_shot_reward"].append(long_shot_bonus)

        return reward, components

    def step(self, action):
        """Overrides step to track certain metrics."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return obs, reward, done, info
