import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides rewards focused on offensive abilities:
    - Accurate shooting.
    - Effective dribbling to evade opponents.
    - Mastery of long and high passes to break defensive lines.

    This implementation emphasizes on:
    1) Shooting close to the goal.
    2) Dribbling with high control.
    3) Effective pass types usage.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.5
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # team 0 is the controlled team
                ball_pos = o['ball'][:2]
                player_pos = o['left_team'][o['active']]
                goal_pos = [1, 0]  # Opponent's goal

                # Distance to opponent's goal for shooting reward.
                dist_to_goal = np.sqrt((goal_pos[0] - ball_pos[0])**2 + (goal_pos[1] - ball_pos[1])**2)
                if dist_to_goal < 0.2:  # shooting close to goal
                    components['shooting_reward'][rew_index] = self.shooting_reward

                # Check dribbling skill by active control
                if o['sticky_actions'][8] == 1 or o['sticky_actions'][9] == 1:  # Sprint or dribble actions
                    components['dribbling_reward'][rew_index] = self.dribbling_reward * np.random.choice([0, 1, 1], p=[0.1, 0.45, 0.45])  # Simulate dribbling effectiveness

                # Passing reward computation
                if o['game_mode'] in {2, 5, 6}:  # Pass movements like GoalKick, ThrowIn, and Penalty
                    # Increasing rewarding for successful long/high passes.
                    for j in range(len(o['right_team'])):
                        opponent_pos = o['right_team'][j]
                        dist_to_opponent = np.sqrt((opponent_pos[0] - player_pos[0])**2 + (opponent_pos[1] - player_pos[1])**2)
                        if dist_to_opponent > 0.3:
                            components['passing_reward'][rew_index] += self.passing_reward

            # Aggregate all rewards
            reward[rew_index] = 1 * components['base_score_reward'][rew_index] + \
                                components['passing_reward'][rew_index] + \
                                components['shooting_reward'][rew_index] + \
                                components['dribbling_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
