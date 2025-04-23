import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful long passes under different match conditions. 
    Each such pass that meets criteria based on distance, accuracy, and game conditions grants extra rewards.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_length_threshold = 0.3
        self.extra_reward_for_pass = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = "State_Custom_Backup"
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0, 0.0]}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if a pass was performed by checking ball ownership changes and long distance.
            if ('ball_owned_team' in o and o['ball_owned_team'] == 1 and  # assuming control by the right team
                    'ball_owned_player' in o and o['ball_owned_player'] != -1):  # the ball is owned by a player
                current_ball_pos = o['ball'][:2]
                prev_step_ball_pos = current_ball_pos - o['ball_direction'][:2]
                dist = np.linalg.norm(current_ball_pos - prev_step_ball_pos)

                # Rewarding only passes longer than a certain threshold, assumed no change of possession
                if dist > self.pass_length_threshold:
                    game_mode = o['game_mode']
                    # Amplify reward if challenging conditions are met (e.g., during a free kick or in normal play)
                    if game_mode in {0, 3}:  # 0 == Normal, 3 == FreeKick
                        components['long_pass_reward'][rew_index] = self.extra_reward_for_pass
                        reward[rew_index] += components['long_pass_reward'][rew_index]

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
