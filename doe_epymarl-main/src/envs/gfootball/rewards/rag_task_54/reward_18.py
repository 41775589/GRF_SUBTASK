import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to emphasize teamwork between passers and shooters."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passing_threshold = 0.05
        self.shooting_distance_threshold = 0.2
        self.pass_reward_multiplier = 0.3
        self.setup_reward_multiplier = 0.4
        self.shoot_reward_multiplier = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Nothing to unpack since no state is maintained beyond what is provided by env
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "setup_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            controlled_player_pos = obs['left_team'][obs['active']]
            ball_pos = obs['ball'][:2]  # Get the x, y position of the ball
            dist_to_goal = abs(ball_pos[0] - 1)

            # Check if this agent has possession and can shoot or pass
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                # Check for passes
                if 'active' in obs and obs['sticky_actions'][9]:  # 'action_dribble' is active
                    components['pass_reward'][rew_index] = self.pass_reward_multiplier
                    reward[rew_index] += components['pass_reward'][rew_index]
                
                # Check for favorable shooting conditions
                if dist_to_goal < self.shooting_distance_threshold:
                    components['shoot_reward'][rew_index] = self.shoot_reward_multiplier
                    reward[rew_index] += components['shoot_reward'][rew_index]
                
                # Reward for setting up a team mate within shooting range
                for teammate_pos in obs['left_team']:
                    if np.linalg.norm(teammate_pos - ball_pos) < self.shooting_distance_threshold:
                        components['setup_reward'][rew_index] = self.setup_reward_multiplier
                        reward[rew_index] += components['setup_reward'][rew_index]
                        break

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
