import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper that encourages team synergy during possession changes,
    focusing on strategic positioning and timely offensive and defensive actions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positional_reward": [0.0] * len(reward),
            "possession_change_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_idx in range(len(reward)):
            o = observation[rew_idx]
            current_pos = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
            
            # Reward players for strategic positioning during potential possession changes
            if o['game_mode'] in [3, 4, 5]:  # free kick, corner, throw-in situations
                # Determine the proximity to the ball for defensive positioning
                ball_pos = o['ball'][:2]  # Consider only x, y
                team_pos = o['left_team' if o['ball_owned_team'] == 0 else 'right_team']
                distances = [euclidean(ball_pos, player_pos) for player_pos in team_pos]
                closest_idx = np.argmin(distances)
                if closest_idx == o['active']:
                    components['positional_reward'][rew_idx] += 0.1

            # Bonus for changing possession proactively with strategic deployment
            if 'ball_owned_team' in o:
                prev_ownership = self.position_rewards.get(rew_idx, {'ball_owned_team': None})['ball_owned_team']
                if o['ball_owned_team'] != prev_ownership and o['ball_owned_team'] != -1:    
                    components["possession_change_reward"][rew_idx] += 0.3

            reward[rew_idx] += components['positional_reward'][rew_idx] + components["possession_change_reward"][rew_idx]
            self.position_rewards[rew_idx] = {
                "ball_owned_team": o['ball_owned_team']
            }
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Tracking used of sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['positional_rewards'] = self.position_rewards
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards = from_pickle.get('positional_rewards', {})
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
