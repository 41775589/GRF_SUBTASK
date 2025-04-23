import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides rewards focused on enhancing collaborative plays between shooters and passers.
    This wrapper encourages players to make successful passes within the vicinity of the goal to create ideal scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_threshold = 0.15  # Threshold distance for considering a pass to count towards the reward
        self.goal_x_coord = 1       # x-coordinate for the opponent's goal
        self.collaborative_reward = 0.2  # Reward for collaborative efforts (passing near the goal)
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
        # Extract observations form the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "collaborative_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            player_pos = o['left_team'][o['active']]
            ball_pos = o['ball'][:2]

            # Check if any player from team has the ball and is close to the opponent's goal
            if o['ball_owned_team'] == 0:  # Team 0 is the left team
                # Calculate distance from the ball to the goal
                distance_to_goal = np.abs(ball_pos[0] - self.goal_x_coord)

                # Check if last action was a pass (assuming the presence of 'action_pass' or similar in obs)
                if 'action_pass' in o['sticky_actions'] and o['sticky_actions']['action_pass']:
                    # Reward collaborative pass if it's within the pass threshold distance to the opponent's goal
                    if distance_to_goal < self.pass_threshold:
                        components["collaborative_reward"][idx] = self.collaborative_reward
                        reward[idx] += self.collaborative_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
