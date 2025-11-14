from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial ball position near our defending area to simulate clearing high balls
    builder.SetBallPosition(-0.6, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # The trained agent positioned centrally in defense to practice high balls and clearances
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Place opposing forwards to challenge the agent in aerial duels
    builder.AddPlayer(-0.65, 0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.65, -0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_LM)  # Midfielder to send in crosses
