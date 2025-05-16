import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward system by recognizing and rewarding effective close-range defensive 
    actions, specifically sliding tackles that successfully block shots or quick dribbles near the agent's goal area.
    """

    def __init__(self, env):
        super().__init__(env)
        self._blocked_shots_reward = 5.0  # Increased reward for blocking shots
        self._goal_area_threshold = 0.2   # Distance threshold defining the "close-range" near the goal
        self._proximity_penalty = -2.0    # Penalty for unnecessary sliding far from the goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._slide_action_index = 8      # Index for 'Sliding' in sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'], dtype=int)
        return from_pickle

    def reward(self, reward):
        """Adjusts the reward based on the successful execution of slide actions near the goal area when defensively critical."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "slide_reward": [0.0]}

        player_pos = observation[0]['left_team'][observation[0]['active']]
        goal_pos = [-1, 0]  # Left goal position
        distance_to_goal = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))

        component_messages = []

        if distance_to_goal < self._goal_area_threshold:
            if observation[0]['sticky_actions'][self._slide_action_index]:
                components["slide_reward"][0] += self._blocked_shots_reward
                reward[0] += components["slide_reward"][0]
                component_messages.append(f"Slide block near goal rewarded: {self._blocked_shots_reward}")
            else:
                components["slide_reward"][0] += self._proximity_penalty
                reward[0] += components["slide_reward"][0]
                component_messages.append(f"Missed slide opportunity, penalty: {self._proximity_penalty}")
        else:
            if observation[0]['sticky_actions'][self._slide_action_index]:
                # Applying slide action far from the goal attracts a penalty
                components["slide_reward"][0] += self._proximity_penalty
                reward[0] += components["slide_reward"][0]
                component_messages.append(f"Unnecessary sliding far from goal, penalty: {self._proximity_penalty}")
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add detailed reward components to info for debugging
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_value

        return observation, reward, done, info
