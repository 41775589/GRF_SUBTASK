import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances offensive strategies by promoting agile movements, strategic positioning, and coordinated passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.2  # Reward for making successful passes towards the opponent's half
        self.positioning_reward = 0.1  # Reward for maintaining strategic positioning

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        stored = from_pickle.get('CheckpointRewardWrapper', {'sticky_actions_counter': np.zeros(10, dtype=int)})
        self.sticky_actions_counter = stored.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": np.zeros_like(reward),
            "positioning_reward": np.zeros_like(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Assuming the team "1" is the controlled team
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0: # Ball in opponent's half
                if 'game_mode' in o and o['game_mode'] == 1:  # Assuming game mode 1 indicates successful pass
                    components["passing_reward"][rew_index] = self.pass_reward
                
                # Positioning reward calculation based on proximity to the center forward position
                if 'left_team_roles' in o:
                    target_position = [1, 0]  # Target coordinate (center forward position)
                    player_pos = o['left_team'][o['active']]
                    distance = np.linalg.norm(np.array(player_pos) - np.array(target_position))
                    components["positioning_reward"][rew_index] += self.positioning_reward * (1 - distance)

            # Compute total reward for current observation
            reward[rew_index] += components["passing_reward"][rew_index] + components["positioning_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
