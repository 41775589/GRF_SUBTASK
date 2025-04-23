import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds specialized rewards for encouraging quick transitions 
    from defense to attack with long, accurate passes.
    """
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
        components = {
            "base_score_reward": reward.copy(),
            "counterattack_initiation_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_position = o['ball']
            current_ball_owner_team = o['ball_owned_team']
            game_mode = o['game_mode']

            if game_mode != 0:  # Not in normal play
                continue

            if current_ball_owner_team == o['left_team']:  # Ball is with the left team (our team)
                ball_x_position = current_ball_position[0]

                if ball_x_position < 0:  # Ball is in the defensive half
                    pass_quality = self.assess_pass_quality(o)
                    components["counterattack_initiation_reward"][rew_index] = pass_quality
                    reward[rew_index] += 2 * pass_quality  # High reward for good passes from defense

        return reward, components

    def assess_pass_quality(self, observation):
        """
        Assess the quality of the pass based on the distance and accuracy towards a forward teammate.
        """
        ball_position = observation['ball'][0:2]
        pass_target_quality = 0.0
        for player_position in observation['right_team']:
            # Calculate distance and direction from ball to each attacker in front
            distance = np.linalg.norm(ball_position - player_position)
            if distance > 0.5:  # Consider it a 'long' pass
                direction_similarity = self.calculate_direction_similarity(ball_position, player_position, observation['ball_direction'][0:2])
                pass_target_quality = max(pass_target_quality, direction_similarity - 0.5 * distance)  # Balance between accuracy and distance

        return pass_target_quality

    def calculate_direction_similarity(self, from_pos, to_pos, ball_direction):
        """
        Calculate directional similarity between the actual pass direction and the optimal pass direction.
        """
        ideal_direction = to_pos - from_pos
        ideal_direction /= np.linalg.norm(ideal_direction)  # Normalize
        ball_direction /= np.linalg.norm(ball_direction)   # Normalize
        similarity = np.dot(ideal_direction, ball_direction)
        return max(similarity, 0)  # Similarity should be non-negative

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
