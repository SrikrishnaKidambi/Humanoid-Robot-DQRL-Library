import os
import numpy as np
import time

from humanoid_library import HumanoidWalkEnv

"""
Manual Walking Test - Makes the humanoid walk using scripted torques

Joint mapping (9 joints):
0: right_hip_y (forward/back)
1: right_knee
2: left_hip_y (forward/back)
3: left_knee
4: right_shoulder1 (arm swing)
5: right_elbow
6: left_shoulder1 (arm swing)
7: left_elbow
8: abdomen_y (spine)
"""

def stabilization_phase():
    """Initial phase to let humanoid settle and get balanced"""
    actions = []
    # Hold upright position for 50 steps before walking
    for _ in range(50):
        action = [
            0.0,   # right_hip: neutral
            0.3,   # right_knee: slight extend to support
            0.0,   # left_hip: neutral
            0.3,   # left_knee: slight extend to support
            0.0,   # right_shoulder: neutral
            0.0,   # right_elbow
            0.0,   # left_shoulder: neutral
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean to stay upright
        ]
        actions.append(action)
    return actions


def simple_walking_pattern():
    """Simplified alternating pattern with backward spine lean"""
    cycle = []
    
    # Right leg forward phase (15 steps)
    for _ in range(15):
        action = [
            -0.8,  # right_hip: forward (negative based on your joint convention)
            -0.4,  # right_knee: bend
            0.6,   # left_hip: backward (push)
            0.8,   # left_knee: extend (push off)
            -0.4,  # right_shoulder: back
            0.0,   # right_elbow
            0.4,   # left_shoulder: forward
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean to prevent toppling
        ]
        cycle.append(action)
    
    # Left leg forward phase (15 steps)
    for _ in range(15):
        action = [
            0.6,   # right_hip: backward (push)
            0.8,   # right_knee: extend (push off)
            -0.8,  # left_hip: forward
            -0.4,  # left_knee: bend
            0.4,   # right_shoulder: forward
            0.0,   # right_elbow
            -0.4,  # left_shoulder: back
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean
        ]
        cycle.append(action)
    
    return cycle


def walking_gait_sequence():
    """Complex walking pattern with backward lean"""
    cycle = []
    
    # PHASE 1: Right leg swing forward, left leg pushes
    for _ in range(10):
        action = [
            -0.6,  # right_hip: forward (FLIPPED)
            -0.3,  # right_knee: bend
            0.5,   # left_hip: backward (FLIPPED)
            0.7,   # left_knee: extend/push
            -0.3,  # right_shoulder: back
            0.0,   # right_elbow
            0.3,   # left_shoulder: forward
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean
        ]
        cycle.append(action)
    
    # PHASE 2: Right leg plants, transition
    for _ in range(5):
        action = [
            -0.2,  # right_hip: slow forward swing (FLIPPED)
            0.5,   # right_knee: extend for landing
            0.3,   # left_hip: start backward (FLIPPED)
            -0.6,  # left_knee: bend to lift
            -0.2,  # right_shoulder
            0.0,   # right_elbow
            0.2,   # left_shoulder
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean
        ]
        cycle.append(action)
    
    # PHASE 3: Left leg swing forward, right leg pushes
    for _ in range(10):
        action = [
            0.5,   # right_hip: backward (FLIPPED)
            0.7,   # right_knee: extend/push
            -0.6,  # left_hip: forward (FLIPPED)
            -0.3,  # left_knee: bend
            0.3,   # right_shoulder: forward
            0.0,   # right_elbow
            -0.3,  # left_shoulder: back
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean
        ]
        cycle.append(action)
    
    # PHASE 4: Left leg plants, transition
    for _ in range(5):
        action = [
            0.3,   # right_hip: start backward (FLIPPED)
            -0.6,  # right_knee: bend to lift
            -0.2,  # left_hip: slow forward swing (FLIPPED)
            0.5,   # left_knee: extend for landing
            0.2,   # right_shoulder
            0.0,   # right_elbow
            -0.2,  # left_shoulder
            0.0,   # left_elbow
            0.0   # spine: BACKWARD lean
        ]
        cycle.append(action)
    
    return cycle


def test_manual_walking(env, num_cycles=10, pattern='simple', with_stabilization=True):
    """
    Test manual walking with predefined torque sequences
    
    Args:
        env: HumanoidWalkEnv instance
        num_cycles: Number of walking cycles to perform
        pattern: 'simple' or 'complex'
        with_stabilization: Whether to stabilize first before walking
    """
    
    print(f"\n{'='*60}")
    print(f"Testing Manual Walking - Pattern: {pattern}")
    print(f"{'='*60}\n")
    
    # Get walking pattern
    if pattern == 'simple':
        cycle = simple_walking_pattern()
    else:
        cycle = walking_gait_sequence()
    
    # Add stabilization at the beginning
    if with_stabilization:
        stabilization = stabilization_phase()
        print(f"Stabilization phase: {len(stabilization)} steps")
    else:
        stabilization = []
    
    print(f"Walking cycle length: {len(cycle)} steps")
    total_planned = len(stabilization) + len(cycle) * num_cycles
    print(f"Total steps planned: {total_planned}")
    print(f"\nStarting in 2 seconds...\n")
    time.sleep(2)
    
    # Reset environment
    obs, info = env.reset()
    
    total_reward = 0
    step_count = 0
    start_pos = obs[0]  # initial x position
    
    # Execute stabilization first
    if with_stabilization:
        print("🔄 STABILIZATION PHASE")
        for action in stabilization:
            action_array = np.array(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action_array)
            total_reward += reward
            step_count += 1
            
            if step_count % 20 == 0:
                x_pos, y_pos, z_pos = obs[0], obs[1], obs[2]
                print(f"  Step {step_count:4d} | Height: {z_pos:.3f}m | Stable")
            
            if terminated or truncated:
                print(f"❌ Failed during stabilization at step {step_count}")
                return 0, total_reward
            
            if env.render_mode == "human":
                time.sleep(0.01)
        
        print(f"✅ Stabilization complete! Starting walking...\n")
        start_pos = obs[0]  # Reset start position after stabilization
        total_reward = 0  # Reset reward counter for walking phase
    
    # Execute walking cycles
    for cycle_num in range(num_cycles):
        print(f"\n🚶 Cycle {cycle_num + 1}/{num_cycles}")
        
        for step_in_cycle, action in enumerate(cycle):
            action_array = np.array(action, dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action_array)
            
            total_reward += reward
            step_count += 1
            
            # Print progress every 10 steps
            if step_count % 10 == 0:
                x_pos, y_pos, z_pos = obs[0], obs[1], obs[2]
                distance = x_pos - start_pos
                print(f"  Step {step_count:4d} | Pos: ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}) | "
                      f"Distance: {distance:.3f}m | Reward: {total_reward:.2f}")
            
            # Check if fallen
            if terminated:
                print(f"\n❌ TERMINATED at step {step_count}!")
                print(f"   Reason: Humanoid fell (height = {obs[2]:.3f}m)")
                break
            
            if truncated:
                print(f"\n⏱️  TRUNCATED at step {step_count} (max steps reached)")
                break
            
            if env.render_mode == "human":
                time.sleep(0.01)
        
        if terminated or truncated:
            break
    
    # Final statistics
    final_x = obs[0]
    total_distance = final_x - start_pos
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total steps taken: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Distance traveled: {total_distance:.3f}m")
    if step_count > 0:
        print(f"Average reward/step: {total_reward/step_count:.4f}")
    print(f"Final height: {obs[2]:.3f}m")
    
    # Success metrics
    if total_distance > 0.5:
        print(f"✅ SUCCESS: Walked forward!")
    elif total_distance > 0:
        print(f"⚠️  PARTIAL: Some forward progress")
    else:
        print(f"❌ FAILED: Moved backward or stayed in place")
    
    print(f"{'='*60}\n")
    
    return total_distance, total_reward


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    
    print("\n" + "="*70)
    print("        MANUAL WALKING TEST SUITE")
    print("="*70)
    
    # Create environment with rendering
    print("\n🚀 Initializing environment with rendering...")
    env = HumanoidWalkEnv(render_mode="human")
    
    # Test with stabilization first
    print("\n🚶 Testing SIMPLE walking pattern (with stabilization)...")
    test_manual_walking(env, num_cycles=5, pattern='simple', with_stabilization=True)
    
    # Uncomment to test complex pattern
    # print("\n🚶 Testing COMPLEX walking pattern (with stabilization)...")
    # test_manual_walking(env, num_cycles=5, pattern='complex', with_stabilization=True)
    
    env.close()
    print("\n✅ Test complete!")