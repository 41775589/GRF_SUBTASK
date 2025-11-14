from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.7, -0.28)  # Long ball challenge from right side towards the left

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper to ensure that there is no score directly
    builder.AddPlayer(0.40, -0.25, e_PlayerRole_LB)  # Left Back - the trained agent specializing in defense

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opposing Goalkeeper
    builder.AddPlayer(0.70, -0.30, e_PlayerRole_CF)  # Attacker close to ball to initiate plays

    # The scenario features the trained left-back in a position to defend against a forward
    # making long ball advancements and utilizing skills like sliding tackles and controlling
    # the ball to stop dribbles. The choice of positions and simple opposition is designed
    # to focus training on specific defensive actions in wide areas of the field.
