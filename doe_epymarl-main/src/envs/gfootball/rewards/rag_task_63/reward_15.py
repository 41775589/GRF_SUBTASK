import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function for training a goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shots_saved = 0
        self.punts_cleared = 0
        self.communications = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shots_saved = 0
        self.punts_cleared = 0
        self.communications = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        to_pickle['CheckpointRewardWrapper']['shots_saved'] = self.shots_saved
        to_pickle['CheckpointRewardWrapper']['punts_cleared'] = self.punts_cleared
        to_pickle['CheckpointRewardWrapper']['communications'] = self.communications
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.shots_saved = state['CheckpointRewardWrapper'].get('shots_saved', 0)
        self.punts_cleared = state['CheckpointRewardWrapper'].get('punts_cleared', 0)
        self.communications = state['CheckpointRewardWrapper'].get('communications', 0)
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Decompose the reward components
        components = {
            "base_score_reward": reward.copy(),
            "shot_stop_reward": [0.0] * len(reward),
            "clearance_reward": [0.0] * len(reward),
            "communication_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        # ---------- Reward for shot stopping ----------
        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0:  # Penalty scenario
                # Reward goalkeeper for saving shots
                if reward[rew_index] == 0:
                    components["shot_stop_reward"][rew_index] = 5.0
                    reward[rew_index] += components["shot_stop_reward"][rew_index]
                    self.shots_saved += 1
        
        # ---------- Reward for clearing punts ----------
        for rew_index, o in enumerate(observation):
            if o['game_mode'] in [3, 4] and o['ball_owned_team'] == 0:  # Free-kick or corner scenarios
                ball_pos = o['ball'][0]  # X position of the ball
                if ball_pos < -0.8:  # Ball is close to the goalkeeper
                    components["clearance_reward"][rew_index] = 3.0
                    reward[rew_index] += components["clearance_reward"][rew_index]
                    self.punts_cleared += 1
        
        # ---------- Reward for effective communication ----------
        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # Our team has the ball
                player_pos = o['left_team'][o['active']]  # Position of the active player
                if player_pos[0] < -0.5:  # Active player is in the defensive half
                    players_behind_ball = sum(p[0] < player_pos[0] for p in o['left_team'])
                    if players_behind_ball >= 3:  # At least three players behind the ball
                        components["communication_reward"][rew_index] = 2.0
                        reward[rew_index] += components["communication_reward"][rew_index]
                        self.communications += 1

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
            for i, active in enumerate(agent_obs['sticky_actions']):
                if active:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
