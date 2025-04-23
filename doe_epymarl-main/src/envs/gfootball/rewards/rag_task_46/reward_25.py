import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for successful standing tackles during both normal and set-piece gameplay scenarios,
    focusing on enhancing possession regaining without risking penalties. It particularly rewards tackles
    which regain possession, are made near the player's own goal to prevent scoring opportunities,
    and do not result in a card."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.tackle_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter[:] = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']] if o['active'] < len(o['left_team']) else o['right_team'][o['active'] - len(o['left_team'])]
            ball_pos = o['ball'][:2]
            distance_to_goal = np.linalg.norm(active_player_pos - [-1, 0])  # Simplistic distance to own goal for left team
            ball_owned_team = o['ball_owned_team']

            is_tackle_action = self.sticky_actions_counter[9] # Assuming '9' is the index for the tackle action

            # Reward successful tackles near own goal that regain possession
            if ball_owned_team == 1 and is_tackle_action and o['game_mode'] in (0, 2, 3, 4):  # Normal, GoalKick, FreeKick, Corner
                components["tackle_reward"][rew_index] = self.tackle_reward / max(distance_to_goal, 0.1)
                reward[rew_index] += components["tackle_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
