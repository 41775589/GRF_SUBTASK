import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on offensive strategy, team coordination, and player positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_threshold = 0.1
        self.shoot_threshold = 0.3
        self.position_reward = 0.05
        self.pass_count = 0
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_count'] = self.pass_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_count = from_pickle['pass_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Iterate through observations
        for i in range(len(reward)):
            player_obs = observation[i]

            # Rewards for passing strategy
            if 'sticky_actions' in player_obs and player_obs['sticky_actions'][9]:  # action_dribble
                components['passing_reward'][i] += self.pass_threshold
                self.pass_count += 1
            
            # Rewards for shooting strategy
            if 'sticky_actions' in player_obs and (player_obs['sticky_actions'][5] or  # action_bottom_right
                                                  player_obs['sticky_actions'][3]):  # action_top_right
                # Scale reward by distance to goal
                distance_to_goal = np.abs(player_obs['ball'][0])
                components['shooting_reward'][i] += self.shoot_threshold / max(distance_to_goal, 0.1)
            
            # Rewards for positioning
            team_pos = player_obs['left_team'] if player_obs['ball_owned_team'] == 0 else player_obs['right_team']
            opponent_goal = 1 if player_obs['ball_owned_team'] == 0 else -1
            for teammate in team_pos:
                if np.sign(teammate[0]) == np.sign(opponent_goal):  # Same side as opponent goal
                    components['positioning_reward'][i] += self.position_reward

            reward[i] += components['passing_reward'][i] + components['shooting_reward'][i] + components['positioning_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counters
        for agent_obs in observation:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_status
                info[f"sticky_actions_{i}"] = action_status
        
        return observation, reward, done, info
