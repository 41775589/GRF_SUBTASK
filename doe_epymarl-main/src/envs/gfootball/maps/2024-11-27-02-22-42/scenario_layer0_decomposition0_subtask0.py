from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.4, 0.0)  # Position the ball on the opponent's half to focus on offensive plays

    # Setup the team on the left (controlled players focused on offensive subtask)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper (uncontrollable)
    builder.AddPlayer(0.1, 0.0, e_PlayerRole_CF)  # A center forward to practice shots
    builder.AddPlayer(0.1, 0.2, e_PlayerRole_AM)  # Attacking midfielder to practice dribbling and precise passes

    # Setup the team on the right (opposing team with minimal defense setup to challenge but not overwhelm)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opposing goalkeeper
    builder.AddPlayer(-0.3, 0.1, e_PlayerRole_CB)  # Single central back to apply defensive pressure
    builder.AddPlayer(-0.3, -0.1, e_PlayerRole_CB)  # Another central back to simulate realistic game scenarios
