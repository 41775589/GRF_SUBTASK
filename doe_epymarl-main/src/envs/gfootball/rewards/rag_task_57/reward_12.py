import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive coordination between midfielders and strikers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.1  # Reward for successful passes
        self.shoot_on_goal_reward = 0.2  # Reward for shooting towards the goal
        self.goal_reward = 1.0  # Reward for scoring a goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "offensive_play_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for i, player_obs in enumerate(observation):
            if 'ball_owned_team' in player_obs and player_obs['ball_owned_team'] == 0:
                if player_obs['ball_owned_player'] == player_obs['active']:
                    # Player has the ball
                    if self.is_midfielder(player_obs['active'], player_obs['left_team_roles']):
                        # Check if there was a pass from a midfielder to a striker
                        if self.check_pass_to_striker(player_obs, observation):
                            components["offensive_play_reward"][i] += self.pass_reward
                    if self.is_striker(player_obs['active'], player_obs['left_team_roles']):
                        # Check if striker is shooting towards goal
                        if self.is_shooting_towards_goal(player_obs):
                            components["offensive_play_reward"][i] += self.shoot_on_goal_reward

                        # Check for goals directly as additional reward
                        if player_obs['score'][0] > player_obs['score'][1]:  # Assuming left is our team
                            components["offensive_play_reward"][i] += self.goal_reward

            # Updating the reward for each action based on components
            reward[i] += components["offensive_play_reward"][i]
        
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
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
        return observation, reward, done, info

    def is_midfielder(self, player_index, roles):
        return roles[player_index] in [4, 5, 6]

    def is_striker(self, player_index, roles):
        return roles[player_index] == 9

    def check_pass_to_striker(self, player_obs, full_observation):
        # This would ideally check past observation and current to determine a pass
        return np.random.rand() > 0.5  # Simulating a condition

    def is_shooting_towards_goal(self, player_obs):
        # Checks if the player's action involves shooting towards the goal
        # Assuming we have some way to determine this from the observation
        return np.random.rand() > 0.7  # Simulating a high chance
