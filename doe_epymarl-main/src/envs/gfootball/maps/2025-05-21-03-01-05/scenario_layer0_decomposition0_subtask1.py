from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.2, -0.2)  # Start with the ball on the left flank

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.7, -0.35, e_PlayerRole_LB)  # Left-back player
    builder.AddPlayer(-0.6, -0.3, e_PlayerRole_LM)  # Left midfielder (Controllable agent)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # The opposing goalkeeper
    builder.AddPlayer(-0.6, -0.25, e_PlayerRole_RB)  # The right back trying to stop the counter-attack
