import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper designed to enhance the training of agents in defensive scenarios, 
    specifically focusing on effectively using 'Sliding' to block shots and quick dribbles 
    close to their own goal in simulated soccer matches.
    """

    def __init__(self, env):
        super().__init__(env)
        self._slide_action_index = 9  # Assuming the 'Sliding' action is at index 9
        self._reward_for_successful_slide = 2.0
        self._penalty_for_missed_slide = -0.5
        self._penalty_for_unnecessary_slide = -0.3
        self._close_range_threshold = 0.15  # Close range threshold to the goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "slide_reward": [0.0]}

        ball_pos = observation[0]['ball'][:2]
        player_pos = observation[0]['left_team'][observation[0]['active']]
        goal_pos = [-1, 0]  # Assuming a standard football field with left goal at x = -1

        # Calculate player distance to the goal and to the ball
        distance_to_goal = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))
        distance_to_ball = np.linalg.norm(np.array(player_pos) - np.array(ball_pos))

        if distance_to_goal < self._close_range_threshold:
            if observation[0]['sticky_actions'][self._slide_action_index]:
                if distance_to_ball < 0.1:
                    # Player performed a slide at close range and near the ball
                    components["slide_reward"][0] += self._reward_for_successful_slide
                    reward[0] += components["slide_reward"][0]
                else:
                    # Slide executed far from ball
                    components["slide_reward"][0] += self._penalty_for_missed_slide
                    reward[0] += components["slide_reward"][0]
            else:
                # Potential for sliding at close range, but no action taken
                components["slide_reward"][0] += self._penalty_for_missed_slide
                reward[0] += components["slide_reward"][0]
        else:
            if observation[0]['sticky_actions'][self._slide_action_index]:
                # Slide far from goal, classified as unnecessary
                components["slide_reward"][0] += self._penalty_for_unnecessary_slide
                reward[0] += components["slide_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_value

        return observation, reward, done, info
