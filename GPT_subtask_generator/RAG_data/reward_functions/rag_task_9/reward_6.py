import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage passing, shooting, dribbling, and creating scoring opportunities."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward modifiers for different behaviors
        self.pass_reward = 0.1
        self.shot_reward = 0.2
        self.dribble_reward = 0.05
        self.sprint_reward = 0.03
        # Track whether each kind of action has been taken at least once
        self.action_taken = {
            'pass': False,
            'shot': False,
            'dribble': False,
            'sprint': False
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.action_taken = {key: False for key in self.action_taken}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "pass_reward": 0.0,
                      "shot_reward": 0.0,
                      "dribble_reward": 0.0,
                      "sprint_reward": 0.0}

        if observation is None:
            return reward, components

        ball_owned_team = observation['ball_owned_team']
        
        # Access sticky actions and controlled players directly
        sticky_actions = observation['sticky_actions']
        active_player = observation['active']

        # Validate if the team has the ball control
        if ball_owned_team == 1:  # Assuming our agent is on the right team which is indexed as 1
            active_actions = sticky_actions[:, active_player]
            # Checking for pass type actions
            if (active_actions[0] == 1 or active_actions[3] == 1) and not self.action_taken['pass']:
                reward += self.pass_reward
                components["pass_reward"] += self.pass_reward
                self.action_taken['pass'] = True

            # Checking for shot action
            if active_actions[9] == 1 and not self.action_taken['shot']:
                reward += self.shot_reward
                components["shot_reward"] += self.shot_reward
                self.action_taken['shot'] = True

            # Checking for dribble action
            if active_actions[7] == 1 and not self.action_taken['dribble']:
                reward += self.dribble_reward
                components["dribble_reward"] += self.dribble_reward
                self.action_taken['dribble'] = True

            # Checking for sprint action
            if active_actions[8] == 1 and not self.action_taken['sprint']:
                reward += self.sprint_reward
                components["sprint_reward"] += self.sprint_reward
                self.action_taken['sprint'] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
