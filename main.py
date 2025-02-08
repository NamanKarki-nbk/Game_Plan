
import os
import cv2
import numpy as np
import torch
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Read Video
    video_path = 'input_videos/eagle_4.mp4'
    video_frames = read_video(video_path)

    # Generate a unique stub path
    video_name = os.path.splitext(os.path.basename(video_path))[0]  
    stub_path = f'stubs/{video_name}_track_stubs.pkl'

    # Initialize Tracker
    tracker = Tracker('models/best2.pt')
    
    # Get Tracks (validate or regenerate as needed)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
    
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    
    # View Trasnformer
    frame_height, frame_width, _ = video_frames[0].shape
    view_transformer = ViewTransformer((frame_height, frame_width))
    view_transformer.add_transformed_position_to_tracks(tracks)


    
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams using TeamAssigner
    team_assigner = TeamAssigner(device="cuda" if torch.cuda.is_available() else "cpu", video_path=video_path)

    # Ensure saved team assignments are loaded (avoid recomputation)
    team_assigner.load_team_assignments()

    for frame_num, player_track in enumerate(tracks['players']):
        player_ids = list(player_track.keys())
        player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]

        # Extract and reduce features
        player_crops = team_assigner.extract_player_crops(video_frames[frame_num], player_bboxes, [1.0] * len(player_ids))
        features = team_assigner.extract_features(player_ids, player_crops)
        reduced_features = team_assigner.reduce_dimensionality(features)

        # Assign teams
        labels = team_assigner.assign_teams_by_track_id(player_ids, reduced_features, reassign=(frame_num % 30 == 0))

        for pid, label in zip(player_ids, labels):
            tracks['players'][frame_num][pid]['team'] = label  # ✅ Assign team normally

        # 🔹 Ensure every player has a valid 'team' entry
        if 'team' not in tracks['players'][frame_num][pid]:
            tracks['players'][frame_num][pid]['team'] = "Unknown"  # Default team assignment


    # Save assigned teams for future runs
    team_assigner.save_team_assignments()


    # Assign Ball to Players
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        # ✅ Safely get ball information
        ball_info = tracks['ball'][frame_num] if frame_num < len(tracks['ball']) else {}
        ball_bbox = ball_info.get(1, {}).get("bbox", None) if isinstance(ball_info, dict) else None

        if not ball_bbox:
            print(f"⚠️ Frame {frame_num}: Ball not detected, using last known team control.")
            last_team = team_ball_control[-1] if team_ball_control else "Unknown"
            team_ball_control.append(last_team)
            continue

        # Assign the ball to the closest player
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            if assigned_player in tracks['players'][frame_num]:
                player_data = tracks['players'][frame_num][assigned_player]

                # ✅ Ensure a team is always assigned
                if 'team' not in player_data:
                    print(f"⚠️ Frame {frame_num}: Assigned player {assigned_player} has no team! Assigning default team.")
                    player_data['team'] = 0  # Default team to avoid UI errors

                # ✅ Assign ball possession
                player_data['has_ball'] = True
                team_ball_control.append(player_data['team'])
                print(f"✅ Frame {frame_num}: Player {assigned_player} has ball. Team: {player_data['team']}")
            else:
                print(f"⚠️ Frame {frame_num}: Assigned player {assigned_player} not found in tracking data!")
                last_team = team_ball_control[-1] if team_ball_control else "Unknown"
                team_ball_control.append(last_team)
        else:
            # Maintain previous team possession if no assignment is found
            last_team = team_ball_control[-1] if team_ball_control else "Unknown"
            team_ball_control.append(last_team)

    team_ball_control = np.array(team_ball_control)  # Convert to NumPy array



    
    # Draw Annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks,  team_ball_control)
    
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)



    # Debugging Information
    print(f"Input video frames: {len(video_frames)}")
    print(f"Output video frames: {len(output_video_frames)}")

    # Save Annotated Video
    save_video(output_video_frames,'output_videos/output.avi')

if __name__ == '__main__':
    main()
