from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting initial ball position for the scenario
    builder.SetBallPosition(0.0, 0.0)

    # Configure Left Team (controlled team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CB)  # Player to train passive defenses

    # Configure Right Team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's goalkeeper
    builder.AddPlayer(-0.5, 0.1, e_PlayerRole_CF)  # Opponent forward to challenge our player
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CM)  # Another opponent player coming into play

    # We focus on defensive stances, hence extra opposing players to simulate pressure
