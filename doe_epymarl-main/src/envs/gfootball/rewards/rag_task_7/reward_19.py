import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents specifically for mastering and correctly timing sliding tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.tackle_position_rewards = {
            0.25: 0.05,  # Slightly out of the box defensive tackle
            0.50: 0.1,   # Midfield tackles
            0.75: 0.2,   # Defensive third tackles
            1.00: 0.5    # Vital defensive tackles near the goal
        }
        self.previous_sliding_tackle = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_sliding_tackle = False
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        reward_adjustments = np.zeros(len(reward))

        for i, (o, rew) in enumerate(zip(observation, reward)):
            current_tackle_action = (
                o["sticky_actions"][-2] if 'sticky_actions' in o else 0
            )
            if current_tackle_action and not self.previous_sliding_tackle:
                player_pos = o['right_team'][o['active']]
                y_dist_to_goal = abs(player_pos[1])
                for position, pos_reward in self.tackle_position_rewards.items():
                    if y_dist_to_goal <= position:
                        components.setdefault("tackle_position_reward", [0.0]*len(reward))
                        components["tackle_position_reward"][i] = pos_reward
                        reward_adjustments[i] += pos_reward
                        break

            self.previous_sliding_tackle = current_tackle_action

        total_reward = [r+adj for r, adj in zip(reward, reward_adjustments)]
        return total_reward, components

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
