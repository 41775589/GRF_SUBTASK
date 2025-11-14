from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.5, 0.0)  # Setting ball in an advanced position to facilitate offensive play

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.3, 0.1, e_PlayerRole_CM)  # Center Midfielder acting as a forward-moving playmaker
    builder.AddPlayer(0.5, -0.1, e_PlayerRole_CF)  # Center Forward for aggressive goal scoring

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CB)  # Opponent center back
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_CB)  # Another opponent center back
