import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specialization for close-range attacks with dribbles and precision shooting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_rewards = 0.0
        self.precision_shot_increment = 0.3
        self.dribble_effectiveness_increment = 0.2

    def reset(self):
        """Reset the environment and reward counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_rewards = 0.0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state to allow proper pickling."""
        to_pickle['total_rewards'] = self.total_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Retrieve state from unpickling."""
        from_pickle = self.env.set_state(state)
        self.total_rewards = from_pickle.get('total_rewards', 0.0)
        return from_pickle

    def reward(self, reward):
        """Modify reward based on possession, position and action effectiveness."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward),
                      "precision_shot_reward": 0.0,
                      "dribble_effectiveness_reward": 0.0}

        if observation is None or reward is None:
            return reward

        for index, r in enumerate(reward):
            # Get corresponding observation for each agent
            o = observation[index]

            # Give additional reward for dribbling towards and shooting accurately close to the goal
            if o['ball_owned_team'] == o['active'] and (o['game_mode'] == 0 or o['game_mode'] == 5):
                # Positioning close to the goal
                if o['ball'][0] > 0.7:  # Assuming 1.0 is very close to the opposite goal
                    components["precision_shot_reward"] += self.precision_shot_increment * np.exp(-o['ball'][1]**2)
                # Successfully executing dribbles in opponent's half
                if o['sticky_actions'][9] == 1 and o['ball'][0] > 0:
                    components["dribble_effectiveness_reward"] += self.dribble_effectiveness_increment

            # Update the total reward given to the agent
            reward[index] += components["precision_shot_reward"] + components["dribble_effectiveness_reward"]

        self.total_rewards += sum(reward)
        return reward, components

    def step(self, action):
        """Apply the action, modify the reward, and return the modified observation."""
        observation, reward, done, info = self.env.step(action)
        new_reward, reward_components = self.reward(reward)
        info['final_reward'] = sum(new_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = value
        return observation, new_reward, done, info
