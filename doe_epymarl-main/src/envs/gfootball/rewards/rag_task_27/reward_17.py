import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that emphasizes defensive skills such as interception and responsive positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 2.0  # High reward for successful interceptions
        self.positioning_reward = 0.1  # Incremental reward for good defensive positioning
        self.previous_ball_owned_team = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_owned_team': self.previous_ball_owned_team
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle.get('CheckpointRewardWrapper', {})
        self.previous_ball_owned_team = state_data.get('previous_ball_owned_team', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        current_ball_owned_team = observation[0].get('ball_owned_team', -1)

        for rew_index, o in enumerate(observation):
            # Reward for intercepting the ball
            if self.previous_ball_owned_team != current_ball_owned_team:
                if current_ball_owned_team == 0:
                    components["interception_reward"][rew_index] += self.interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]

            # Reward for maintaining position close to the ball defensively
            ball_position = o.get('ball', [0, 0])
            player_position = o['right_team'][o['active']] if current_ball_owned_team == 1 else o['left_team'][o['active']]
            distance_to_ball = np.linalg.norm(np.array(ball_position[:2]) - np.array(player_position))
            components["positioning_reward"][rew_index] += max(self.positioning_reward / (distance_to_ball + 0.1), 0)

            reward[rew_index] += components["positioning_reward"][rew_index]

        self.previous_ball_owned_team = current_ball_owned_team
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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
