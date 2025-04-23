import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards focused on shot precision in close proximity to the goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.shot_zones = [0.1, 0.2, 0.3]  # Define zones near the goal post to encourage shooting
        self.zone_rewards = [0.2, 0.3, 0.5]  # Encourage shots closer to goal with higher rewards
        self.shot_threshold = 0.08  # How close to the goal line the player needs to be for a shot
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_pos = (0, 0)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_pos = (0, 0)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "precision_shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if near goal and shooting
            shot_zone = None
            for zone, limit in enumerate(self.shot_zones):
                if abs(o['ball'][0]) > (1 - limit):  # x is close to either -1 or 1
                    shot_zone = zone

            if shot_zone is not None:
                # Calculate additional rewards only for appropriate game modes
                if o['game_mode'] == 0:  # Normal gameplay
                    distance = np.sqrt((o['ball'][0] - self.last_ball_pos[0])**2 +
                                       (o['ball'][1] - self.last_ball_pos[1])**2)
                    if distance > self.shot_threshold and o['ball_owned_team'] == 0:  # Assuming team 0 is controlled
                        # Adding rewards depending on how close to the goal the shot was made
                        components['precision_shot_reward'][rew_index] = self.zone_rewards[shot_zone]
                        reward[rew_index] += components['precision_shot_reward'][rew_index]

            # Update the last ball position
            self.last_ball_pos = o['ball'][:2]

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
