import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward related to dribbling, sprinting, and evasion techniques in offensive scenarios.

    This class focuses on rewarding the agent for:
    1. Maintaining control over the ball while sprinting.
    2. Advancing towards the opponent's goal.
    3. Evasive maneuvers around defenders.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To count sticky actions like sprints, dribbles

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_rewards'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['checkpoint_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_reward": [0.0] * len(reward),
                      "advancement_reward": [0.0] * len(reward),
                      "evasion_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, (rew, o) in enumerate(zip(reward, observation)):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Assuming agent's team is '0'
                # Control and sprint reward
                if o['sticky_actions'][8] == 1:  # Sprint is active
                    components['control_reward'][i] = 0.1  # Small reward for sprinting with the ball
                # Advancement reward
                ball_x = o['ball'][0]
                components['advancement_reward'][i] = max(0, ball_x * 0.1)  # Reward increased as ball moves forward
            
            # Evasion reward based on proximity to opponents
            my_player_pos = o['left_team'][o['active']]
            opponents = o['right_team']
            proximities = [np.linalg.norm(my_player_pos - p) for p in opponents]

            # Reward for maintaining distance to the nearest opponent (evasive maneuver success)
            min_proximity = min(proximities)
            components['evasion_reward'][i] = max(0, (0.5 - min_proximity) * 0.2)  # Reward if close to any opponent
            
            # Compile the total reward for this agent
            reward[i] += sum(components[k][i] for k in components)

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
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
