import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a scenario-based reward focusing on passing and shooting training."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        # Define reward components
        for i, o in enumerate(observation):
            # Enhance precision and strategic positioning through ball possession
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:  # If active player has ball possession
                near_goal = 1.0 if o['ball'][0] > 0.5 else 0.0  # Bonus for being in opponent's half
                components.setdefault("possession_bonus", []).append(near_goal)
                reward[i] += near_goal

                # Reward for successful passes or shots, especially near opponent goal
                shoot_direction = np.abs(o['ball_direction'][1]) > 0.1 # indicating a more vertical shoot, closer to goal orientation
                if shoot_direction:
                    components.setdefault("shoot_bonus", []).append(0.5)
                    reward[i] += 0.5

                # Additional reward for strategic passes or clears under pressure
                near_opponents = any([np.linalg.norm(o['ball'] - pos) < 0.1 for pos in o['right_team']])
                if near_opponents:
                    components.setdefault("pressure_pass_bonus", []).append(0.3)
                    reward[i] += 0.3
            
            else:
                components.setdefault("possession_bonus", []).append(0.0)
                components.setdefault("shoot_bonus", []).append(0.0)
                components.setdefault("pressure_pass_bonus", []).append(0.0)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        self.sticky_actions_counter.fill(0)
        if observation:
            for agent_obs in observation:
                for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                    info[f"sticky_actions_{i}"] = action
                    
        return observation, reward, done, info
