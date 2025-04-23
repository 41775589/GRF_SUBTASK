import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive and transition play cues in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.teammate_history_positions = []
        self.opponent_history_positions = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.teammate_history_positions = []
        self.opponent_history_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'teammate_history_positions': self.teammate_history_positions,
            'opponent_history_positions': self.opponent_history_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.teammate_history_positions = from_pickle['CheckpointRewardWrapper'].get('teammate_history_positions', [])
        self.opponent_history_positions = from_pickle['CheckpointRewardWrapper'].get('opponent_history_positions', [])
        return from_pickle

    def reward(self, reward):
        base_score_reward = reward.copy()
        transition_reward = [0.0] * len(reward)
        defense_coordination_reward = [0.0] * len(reward)
        components = {
            "base_score_reward": base_score_reward,
            "transition_reward": transition_reward,
            "defense_coordination_reward": defense_coordination_reward
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        # Defensive coordination reward: incentive for maintaining formation and repelling attacks
        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # Opponent has the ball
                ball_position = np.array(o['ball'][:2])  # only consider x, y
                team_positions = [(p[0], p[1]) for p in o['left_team']]
                opponent_positions = [(p[0], p[1]) for p in o['right_team']]

                # Calculate distances to the ball from all players and opponent players
                team_distances = [np.linalg.norm(np.array(pos) - ball_position) for pos in team_positions]
                opponent_distances = [np.linalg.norm(np.array(pos) - ball_position) for pos in opponent_positions]

                # Reward for close defense (closer than any opponent player to ball)
                if min(team_distances) < min(opponent_distances):
                    defense_coordination_reward[i] += 0.1
                self.teammate_history_positions.append(team_positions)
                self.opponent_history_positions.append(opponent_positions)

            # Transition Reward: efficient movement of the ball from defense to attack
            if o['ball_owned_team'] == 0:  # If our team has the ball
                if len(self.teammate_history_positions) > 1:
                    prev_positions = self.teammate_history_positions[-1]
                    current_positions = [(p[0], p[1]) for p in o['left_team']]
                    movement = [np.linalg.norm(np.array(curr) - np.array(prev)) for curr, prev in zip(current_positions, prev_positions)]
                    if sum(movement) < 0.1 * len(movement):  # Low movement might indicate good positioning
                        transition_reward[i] += 0.05
        
        # Scale the rewards
        final_rewards = [sum(x) for x in zip(base_score_reward, transition_reward, defense_coordination_reward)]
        return final_rewards, components

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
