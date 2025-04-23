import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for close-range attacks, precision in shooting, and dribble effectiveness against goalkeepers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "precision_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage shooting precision when near the goal
            if o['game_mode'] == 6:  # Penalty mode close to the goal
                if o['ball_owned_player'] == o['active']:
                    components["precision_reward"][rew_index] = 0.2

            # Encourage effective dribbling right before shooting
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0: # Own team has the ball
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                distance_to_goal = np.linalg.norm(player_pos - [1, 0])  # Distance to opponent's goal

                # Reward dribbling moves as the agent moves close to the goalkeeper (within 20% of the pitch length)
                if distance_to_goal < 0.2:
                    if o['sticky_actions'][9]:  # Dribble action is active
                        components["precision_reward"][rew_index] += 0.1
                
            reward[rew_index] += sum(components.values())
            
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
