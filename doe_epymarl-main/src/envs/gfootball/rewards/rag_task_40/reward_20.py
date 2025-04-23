import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards to enhance defensive capabilities and prepare for counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Initial reward added to components dictionary
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning": [0.0] * len(reward),
                      "counterattack_readiness": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()

        # Early exit if no observation is found
        if observation is None:
            return reward, components

        # Assuming observation for two agents (defense-specialized agents)
        for agent_index, obs in enumerate(observation):

            # Encourage players to position closer to defensive regions when the ball is with opponent
            if obs['ball_owned_team'] == 1:  # Ball is with the opponent
                # Calculate the distance to the own goal (assuming left side is own goal)
                x_own_goal = -1
                player_x = obs['left_team'][obs['active']][0]  # Get x-coordinate of the active player
                distance_to_own_goal = np.abs(player_x - x_own_goal)

                # Defensive positioning reward boost for being closer to own goal
                defense_reward = max(0.1, 1.0 - distance_to_own_goal) * 0.2  # Scaled and capped between 0 to 0.2
                components['defensive_positioning'][agent_index] = defense_reward
            
            # Encourage counterattacking readiness when possession is regained or maintained
            if obs['ball_owned_team'] == 0:  # Ball is with own team
                # Bonus for facing the right direction, closer to the right side
                if not obs['left_team_roles'][obs['active']] == 0:  # Non-goalkeeper
                    facing_right = obs['left_team_direction'][obs['active']][0] > 0  # Check if moving towards right
                    counterattack_bonus = 0.1 if facing_right else 0
                    components['counterattack_readiness'][agent_index] = counterattack_bonus

            # Calculate final reward for this cycle
            reward[agent_index] += components['defensive_positioning'][agent_index] + components['counterattack_readiness'][agent_index]

        return reward, components

    def step(self, action):
        # Execute action in the environment
        observation, reward, done, info = self.env.step(action)

        # Modify rewards using custom reward function
        reward, components = self.reward(reward)

        # Store final reward in info for debugging
        info["final_reward"] = sum(reward)
        
        # Store components of reward in info for analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
