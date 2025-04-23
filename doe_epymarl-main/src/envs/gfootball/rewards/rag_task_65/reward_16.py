import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards based on shooting and passing 
       precision, strategic positioning and decision-making in a 
       football simulation."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_accuracy_reward = 0.5
        self.shoot_accuracy_reward = 1.0
        self.strategic_positioning_reward = 0.3
        self.decision_making_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset state for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment to pickle."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from unpickled state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the rewards based on passing precision, shooting accuracy,
           strategic positioning, and decisions made by agents."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": list(reward),
            "pass_accuracy_reward": [0.0] * len(reward),
            "shoot_accuracy_reward": [0.0] * len(reward),
            "strategic_positioning_reward": [0.0] * len(reward),
            "decision_making_reward": [0.0] * len(reward),
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Increase reward based on ball control and decisions leading to successful passes or shots
            if o['ball_owned_team'] == 0:  # Assume agent's team is '0'
                if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                    # Check for successful passes in strategic positions
                    if o['game_mode'] == 2 or o['game_mode'] == 5:  # Pass or throw in
                        components['pass_accuracy_reward'][rew_index] = self.pass_accuracy_reward
                    # Check for attempts to score
                    elif o['game_mode'] == 4:  # Shoot or scoring
                        distance_to_goal = abs(o['ball'][0] - 1)  # X position 1 represents opponent's goal on the right
                        components['shoot_accuracy_reward'][rew_index] = self.shoot_accuracy_reward / max(0.1, distance_to_goal)

            # Reward for strategic movements and positioning without the ball
            if 'right_team' in o and o['game_mode'] == 0:  # Normal game mode
                friend_positions = o['right_team']
                ball_position = o['ball'][:2]
                distances = np.sqrt(np.sum((friend_positions - ball_position)**2, axis=1))
                # Closer positions to the ball within threshold could receive a strategic reward
                strategic_positions = distances < 0.2
                if any(strategic_positions):
                    components['strategic_positioning_reward'][rew_index] = self.strategic_positioning_reward

            # Aggregate rewards
            reward[rew_index] += sum([
                components['pass_accuracy_reward'][rew_index],
                components['shoot_accuracy_reward'][rew_index],
                components['strategic_positioning_reward'][rew_index],
                components['decision_making_reward'][rew_index]
            ])

        return reward, components

    def step(self, action):
        """Steps through the environment, applying the new reward system."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Updating sticky actions information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
