import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward component focused on defensive capabilities."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define defense-focused rewards based on ball possession change and defensive actions
        self.own_goal_area_entered = False
        self.opponent_goal_area_entered = False
        self.tackles_made = 0
        self.defensive_play_reward = 0.2  # Reward for successful defensive actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.own_goal_area_entered = False
        self.opponent_goal_area_entered = False
        self.tackles_made = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()  # Base reward from the environment
        defense_reward = [0.0] * len(reward)
        observed_tackles = 0

        if observation is None:
            return reward, {'base_score_reward': base_score_reward, 'defense_reward': defense_reward}

        for i, obs in enumerate(observation):
            if obs['game_mode'] in [3, 4]:  # Game modes that involve defense scenarios (Free kick, Corner)
                if obs['ball_owned_team'] == 0 and (not self.own_goal_area_entered) and self.is_in_goal_area(obs['ball'], own_goal=True):
                    self.own_goal_area_entered = True
                elif obs['ball_owned_team'] == 1 and (not self.opponent_goal_area_entered) and self.is_in_goal_area(obs['ball'], own_goal=False):
                    self.opponent_goal_area_entered = True

            # Check for tackling action
            if obs['ball_owned_team'] == 1 and 'sticky_actions' in obs:
                if obs['sticky_actions'][7]:  # Assume index 7 tracks tackles
                    observed_tackles += 1

        # Calculate defense reward based on the change of ball possessions within the goal areas
        if self.own_goal_area_entered:
            defense_reward = [d + self.defensive_play_reward for d in defense_reward]
            self.own_goal_area_entered = False

        self.tackles_made += observed_tackles
        defense_reward = [d + observed_tackles * 0.1 for d in defense_reward]

        final_reward = [br + dr for br, dr in zip(base_score_reward, defense_reward)]
        return final_reward, {'base_score_reward': base_score_reward, 'defense_reward': defense_reward}

    def is_in_goal_area(self, ball_position, own_goal):
        """ Helper function to determine if the ball is in the goal area, change limits based on own or opponent goal """
        if own_goal:
            return -1 <= ball_position[0] <= -0.8 and -0.22 <= ball_position[1] <= 0.22
        else:
            return 0.8 <= ball_position[0] <= 1 and -0.22 <= ball_position[1] <= 0.22

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Update the info dictionary with reward breakdown
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += int(action_active)
        
        return obs, reward, done, info
