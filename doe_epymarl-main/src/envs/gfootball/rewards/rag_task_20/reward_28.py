import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive strategies in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defining thresholds and rewards for positioning and ball control
        self.positioning_reward = 0.05  # reward for good positioning
        self.shot_on_goal_reward = 0.2  # reward for shooting towards goal
        self.passing_reward = 0.1       # reward for successful passes

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
                      "positioning_reward": [0.0] * len(reward),
                      "shot_on_goal_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        for i, r in enumerate(reward):
            obs = observation[i]

            # Encourage maintaining possession and progressing forward
            if obs['ball_owned_team'] == 1:
                # Check if it's closer to the opponent's goal
                if obs['ball'][0] > 0.5:  # assuming right side is opponent's side
                    components['positioning_reward'][i] = self.positioning_reward

                # Encourage shooting when close to the goal
                if obs['ball'][0] > 0.7 and obs['game_mode'] == 0:
                    components['shot_on_goal_reward'][i] = self.shot_on_goal_reward

                # Add reward for successful passes (change of ball ownership while in possession)
                if ('ball_owned_player' in obs and
                        obs['ball_owned_player'] != obs['active'] and
                        obs['ball_owned_player'] != -1):
                    components['passing_reward'][i] = self.passing_reward

            # Combine the rewards with the existing game rewards
            reward[i] += (components['positioning_reward'][i] +
                          components['shot_on_goal_reward'][i] +
                          components['passing_reward'][i])

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
