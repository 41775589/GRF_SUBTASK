from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)  # Centre Back, controlling player
    builder.AddPlayer(-0.5, 0.15, e_PlayerRole_CM)  # Centre Midfielder
    builder.AddPlayer(-0.5, -0.15, e_PlayerRole_CM)  # Centre Midfielder

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opposing Goalkeeper
    builder.AddPlayer(-0.6, 0.10, e_PlayerRole_CM)  # Opposing Centre Midfielder
    builder.AddPlayer(-0.6, 0.0, e_PlayerRole_CF)  # Opposing Centre Forward
    builder.AddPlayer(-0.6, -0.10, e_PlayerRole_CM)  # Opposing Centre Midfielder
