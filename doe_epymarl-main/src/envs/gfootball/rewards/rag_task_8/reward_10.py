import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for counter-attack initiation and quick decision-making in ball handling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for each player
        self.prev_ball_owned_team = -1  # Track who owned the ball previously

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_owned_team = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter,
            "prev_ball_owned_team": self.prev_ball_owned_team
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = wrapper_state["sticky_actions_counter"]
        self.prev_ball_owned_team = wrapper_state["prev_ball_owned_team"]
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counter_attack_reward": [0.0] * len(reward)}

        current_ball_owned_team = observation[0]['ball_owned_team']

        # Encourages quick passes and control under pressure after gaining possession
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Assuming the agent's team is 0
                # Reward for quick decision making after recovering the ball
                if self.prev_ball_owned_team != 0 and o['ball_owned_team'] == 0:
                    components["counter_attack_reward"][rew_index] = 2.0  # Scaled reward for initiating attack
                    reward[rew_index] += components["counter_attack_reward"][rew_index]

            # Update previous ownership for the next step
            self.prev_ball_owned_team = current_ball_owned_team

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Integrating the reward components into info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
