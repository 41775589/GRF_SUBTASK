from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial ball position near the opposing team's goal to simulate a defense-to-attack transition
    builder.SetBallPosition(0.8, 0.0)

    builder.SetTeam(Team.e_Left)
    # Adding a goalkeeper for the left team
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Adding the single agent trained for sprint and stop-sprint, positioned to initially defend
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)

    builder.SetTeam(Team.e_Right)
    # Adding a goalkeeper for the right team
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Adding an opposing forward to apply offensive pressure
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)
