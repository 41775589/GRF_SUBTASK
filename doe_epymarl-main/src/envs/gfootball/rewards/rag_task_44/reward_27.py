import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for managing ball control under pressure
    using the Stop-Dribble action as a defensive tactic.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_dribble_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the received rewards by adding a reward if the agent chooses to stop dribbling
        under defensive pressure while maintaining control of the ball.
        """

        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i in range(len(reward)):
            agent_obs = observation[i]
            is_dribbling = agent_obs['sticky_actions'][9] == 1  # action_dribble is at the 9th index
            is_sprinting = agent_obs['sticky_actions'][8] == 1  # action_sprint is at the 8th index

            opponent_team = 'right_team' if agent_obs['ball_owned_team'] == 0 else 'left_team'
            team_close = self.compute_proximity(agent_obs['left_team'] if agent_obs['ball_owned_team'] == 1 else agent_obs['right_team'], agent_obs['ball'])
            opponent_close = self.compute_proximity(agent_obs[opponent_team], agent_obs['ball'])

            # Awarding the reward if the player stops dribbling while under defensive pressure
            if is_dribbling and not is_sprinting and opponent_close < team_close:
                reward[i] += self.stop_dribble_reward
                components["stop_dribble_reward"][i] = self.stop_dribble_reward

        return reward, components
    
    @staticmethod
    def compute_proximity(team_positions, ball_pos):
        """
        Computes average proximity of the team to the ball
        """
        distances = np.sqrt(np.sum((team_positions - ball_pos[:2]) ** 2, axis=1))
        return np.mean(distances)

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
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
