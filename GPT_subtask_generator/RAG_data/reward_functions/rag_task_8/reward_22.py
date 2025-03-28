import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for efficient ball handling and quick decision-making for initiating counter-attacks
    after recovering the ball possession.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "counter_attack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, o in enumerate(observation):
            # Check if the ball is just recovered and a counter-attack is possible
            if (o['ball_owned_team'] == 0 and 
                self.sticky_actions_counter[9] > 0):  # dribbling action
                ball_pos = o['ball']
                opponent_goal_pos = 1  # x coordinate of the opponent's goal
                distance_from_goal = opponent_goal_pos - ball_pos[0]  # x coordinate of the ball
                if distance_from_goal > 0:
                    # Reward inversely proportional to the distance to opponent's goal
                    components["counter_attack_reward"][index] = 1.0 / (1 + distance_from_goal)
                    reward[index] += components["counter_attack_reward"][index]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
