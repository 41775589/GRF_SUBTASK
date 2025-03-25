import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a sprint-based reward for improving defensive coverage.
    It encourages players to utilize the sprint action frequently and effectively,
    positioning themselves optimally in relation to the ball and other players.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Count the usage of each sticky action

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_rewarded_sprints'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['checkpoint_rewarded_sprints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward effective sprint usage based on player's positioning and movement.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sprint_action = o['sticky_actions'][8]  # index for sprint action
            
            # Update sprint counter
            if sprint_action:
                self.sticky_actions_counter[8] += 1

            # Calculate a bonus for sprinting in a strategically beneficial manner
            team_pos = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
            opponent_pos = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
            ball_pos = o['ball'][:2]
            
            # Calculate distances from the ball and the closest opponent
            player_pos = team_pos[o['active']]
            dist_to_ball = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))
            dist_to_closest_opponent = np.min(np.linalg.norm(np.array(opponent_pos) - np.array(player_pos), axis=1))

            # Define reward thresholds
            effective_distance_to_ball = 0.2
            effective_distance_to_opponent = 0.3

            # Compute the reward if the player is sprinting effectively
            if sprint_action and dist_to_ball < effective_distance_to_ball and dist_to_closest_opponent > effective_distance_to_opponent:
                sprint_bonus = 0.01  # small reward for staying strategically effective while sprinting
                components['sprint_reward'][rew_index] = sprint_bonus
                reward[rew_index] += sprint_bonus

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
                if action:
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
