import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that incentivizes long-range shooting and effective decision-making
    when shooting from outside the penalty box."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state_from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = state_from_pickle['sticky_actions_counter']
        return state_from_pickle

    def reward(self, reward: list[float]) -> tuple[list[float], dict[str, list[float]]]:
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for idx in range(len(reward)):
            o = observation[idx]
            
            # Assuming that agent is in the 'shooting zone' from outside the penalty box
            # based on X position (roughly half-court towards goal)
            outside_penalty_zone = o['ball'][0] > 0.5 if o['ball_owned_team'] == 0 else o['ball'][0] < -0.5
            
            # Checking if the ball is owned by the agent's team
            if o['ball_owned_team'] == 0 and outside_penalty_zone:
                # Triggering reward for effective long-range shot preparation or attempt
                components.setdefault("long_range_shot_preparation", [0.0, 0.0])
                components["long_range_shot_preparation"][idx] += 0.1  # Incremental reward for preparing/attempting long shots
                reward[idx] += components["long_range_shot_preparation"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
