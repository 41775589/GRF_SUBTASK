import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense defensive and transitional play reward for improving
    defensive coordination and ball distribution effectiveness.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = self.env.get_state(to_pickle)
        return state_info

    def set_state(self, state):
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reactions": [0.0] * len(reward),
                      "transitional_play": [0.0] * len(reward)}

        for agent_index, rew in enumerate(reward):
            o = observation[agent_index]
            ball_owned = o["ball_owned_team"] == 0 # Using 0 assuming '0' is the team ID for agents
            opponent_has_ball = o["ball_owned_team"] == 1
            defending_area = o["active"] in o["left_team"] and ball_owned

            # Defensive play: Checking for successful tackles and interceptions
            if opponent_has_ball and not ball_owned:
                components["defensive_reactions"][agent_index] = 0.1
                reward[agent_index] += 0.1 * components["defensive_reactions"][agent_index]

            # Transitional play: Efficient passing and space creation after gaining possession
            if ball_owned and o["game_mode"] == 0:  # Assuming '0' for normal gameplay mode
                components["transitional_play"][agent_index] = 0.1
                reward[agent_index] += 0.1 * components["transitional_play"][agent_index]

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
            for i, action_activated in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_activated
        return observation, reward, done, info
