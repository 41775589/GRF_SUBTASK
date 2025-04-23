import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes from midfield 
       aimed at creating direct scoring opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward_coefficient = 0.5  # Reward scaling factor for successful high passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_action_counters'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_action_counters', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward),
            "high_pass_reward": np.array([0.0] * 2)
        }

        if observation is None:
            return reward, components 

        assert len(reward) == len(observation), "Mismatch in the number of agents and rewards"

        for index, o in enumerate(observation):
            if o['game_mode'] == 2 and o['ball_owned_team'] == 0:  # 2 corresponds to FreeKick which can resemble a high pass situation
                ball_end_pos = np.array(o['ball']) + np.array(o['ball_direction'])  # Predicted ball position after current movement
                # Check if the ball is moving towards the opponent's goal area
                if 0.2 <= ball_end_pos[0] <= 1 and abs(ball_end_pos[1]) < 0.42:  # Ball within scoring zone in the opponent's half
                    components['high_pass_reward'][index] = self.high_pass_reward_coefficient
                    reward[index] += components['high_pass_reward'][index]

        return list(reward), components 

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)  # Process the reward through the custom function
        info["final_reward"] = sum(reward)  # Capture the final total reward
        # Include individual reward components in the info dictionary for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
