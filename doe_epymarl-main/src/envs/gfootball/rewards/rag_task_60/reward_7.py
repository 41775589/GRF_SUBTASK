import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on defensive player positioning and transitions between movements."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_history = [{}, {}]  # Maintain a history of coordinates for both agents

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_history = [{}, {}]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_positions'] = self.position_history
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_history = from_pickle['CheckpointRewardWrapper_positions']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the accuracy and timing of movement transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "movement_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            previous_pos = self.position_history[i].get('prev_position', o['left_team'][o['active']])
            current_pos = o['left_team'][o['active']]
            movement_vector_length = np.linalg.norm(np.array(previous_pos) - np.array(current_pos))

            # Check for movement transition: player stopped moving or started moving
            if 'previous_moving' in self.position_history[i]:
                if movement_vector_length < 0.01 and self.position_history[i]['previous_moving']:
                    components['movement_transition_reward'][i] = 0.5
                elif movement_vector_length > 0.01 and not self.position_history[i]['previous_moving']:
                    components['movement_transition_reward'][i] = 0.5
            self.position_history[i]['previous_moving'] = movement_vector_length > 0.01

            # Update the activity history
            self.position_history[i]['prev_position'] = current_pos
            
            # Update the reward
            reward[i] += components['movement_transition_reward'][i]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
