from collections import deque
import numpy as np


class WinRateTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.results = deque(maxlen=window_size)

        # 总体统计
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        self.total_episodes = 0

    def add_game_result(self, our_score, opponent_score):
        """添加一场比赛结果"""
        if our_score > opponent_score:
            result = 1.0  # 胜
            self.total_wins += 1
        elif our_score < opponent_score:
            result = 0.0  # 负
            self.total_losses += 1
        else:
            result = 0.5  # 平
            self.total_draws += 1

        self.results.append(result)
        self.total_episodes += 1

    def get_recent_win_rate(self):
        """获取最近 N 场的胜率"""
        if not self.results:
            return 0.0
        return np.mean(self.results)

    def get_overall_win_rate(self):
        """获取总体胜率"""
        if self.total_episodes == 0:
            return 0.0
        return self.total_wins / self.total_episodes

    def get_stats(self):
        """获取完整统计信息"""
        return {
            'recent_win_rate': self.get_recent_win_rate(),
            'overall_win_rate': self.get_overall_win_rate(),
            'total_wins': self.total_wins,
            'total_draws': self.total_draws,
            'total_losses': self.total_losses,
            'total_episodes': self.total_episodes
        }