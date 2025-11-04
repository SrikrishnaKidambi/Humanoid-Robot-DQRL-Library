import os
import numpy as np
import time

# Use your library import structure
from humanoid_library import HumanoidWalkEnv

"""
Manual Walking Test - Makes the humanoid walk using scripted torques

Joint mapping (9 joints) - YOUR CURRENT SETUP:
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

def walking_gait_sequence():
    """
    Returns a sequence of actions that simulate a walking gait.
    Each action is [9 torques] corresponding to the joint mapping above.
    
    Walking cycle breakdown:
    1. Right leg forward (left leg pushes)
    2. Right leg plants, left leg lifts
    3. Left leg forward (right leg pushes)
    4. Left leg plants, right leg lifts
    """
    
    cycle = []
    
    # === PHASE 1: Right leg swing forward, left leg pushes back ===
    # Duration: 10 steps
    for _ in range(10):
        action = [
            0.6,   # right_hip_y: swing forward (positive = forward)
            -0.3,  # right_knee: slight bend while swinging
            -0.5,  # left_hip_y: push back (negative = backward)
            0.7,   # left_knee: extend to push off ground
            -0.3,  # right_shoulder: swing back (opposite to leg)
            0.0,   # right_elbow: neutral
            0.3,   # left_shoulder: swing forward
            0.0,   # left_elbow: neutral
            0.0    # abdomen: keep straight
        ]
        cycle.append(action)
    
    # === PHASE 2: Right leg plants, transition ===
    # Duration: 5 steps
    for _ in range(5):
        action = [
            0.2,   # right_hip_y: slow down forward swing
            0.5,   # right_knee: extend for landing
            -0.3,  # left_hip_y: start lifting
            -0.6,  # left_knee: bend to lift foot
            -0.2,  # right_shoulder: continue back swing
            0.0,   # right_elbow
            0.2,   # left_shoulder: continue forward
            0.0,   # left_elbow
            0.0    # abdomen
        ]
        cycle.append(action)
    
    # === PHASE 3: Left leg swing forward, right leg pushes back ===
    # Duration: 10 steps
    for _ in range(10):
        action = [
            -0.5,  # right_hip_y: push back
            0.7,   # right_knee: extend to push
            0.6,   # left_hip_y: swing forward
            -0.3,  # left_knee: slight bend while swinging
            0.3,   # right_shoulder: swing forward (opposite to leg)
            0.0,   # right_elbow
            -0.3,  # left_shoulder: swing back
            0.0,   # left_elbow
            0.0    # abdomen
        ]
        cycle.append(action)
    
    # === PHASE 4: Left leg plants, transition ===
    # Duration: 5 steps
    for _ in range(5):
        action = [
            -0.3,  # right_hip_y: start lifting
            -0.6,  # right_knee: bend to lift foot
            0.2,   # left_hip_y: slow down forward swing
            0.5,   # left_knee: extend for landing
            0.2,   # right_shoulder: continue forward
            0.0,   # right_elbow
            -0.2,  # left_shoulder: continue back swing
            0.0,   # left_elbow
            0.0    # abdomen
        ]
        cycle.append(action)
    
    return cycle


def simple_walking_pattern():
    """
    Simplified alternating pattern - easier to debug
    """
    cycle = []
    
    # Right leg forward phase (15 steps)
def simple_walking_pattern():
    cycle = []
    
    # Right leg forward phase (15 steps)
    for _ in range(15):
        action = [
            -0.8,  # right_hip: FLIP SIGN (was 0.8)
            -0.4,  # right_knee: bend
            0.6,   # left_hip: FLIP SIGN (was -0.6)
            0.8,   # left_knee: extend
            -0.4,  # right_shoulder: back
            0.0,   # right_elbow
            0.4,   # left_shoulder: forward
            0.0,   # left_elbow
            0.0    # spine
        ]
        cycle.append(action)
    
    # Left leg forward phase (15 steps)
    for _ in range(15):
        action = [
            0.6,   # right_hip: FLIP SIGN (was -0.6)
            0.8,   # right_knee: extend
            -0.8,  # left_hip: FLIP SIGN (was 0.8)
            -0.4,  # left_knee: bend
            0.4,   # right_shoulder: forward
            0.0,   # right_elbow
            -0.4,  # left_shoulder: back
            0.0,   # left_elbow
            0.0    # spine
        ]
        cycle.append(action)
    
    return cycle


def extreme_walking_pattern():
    """
    VERY exaggerated movements - to see if ANYTHING happens
    """
    cycle = []
    
    # Right leg MAXIMUM forward
    for _ in range(20):
        action = [
            1.0,   # right_hip: MAX forward
            -0.8,  # right_knee: big bend
            -1.0,  # left_hip: MAX back
            1.0,   # left_knee: MAX extend
            -0.8,  # right_shoulder: big back swing
            0.0,   # right_elbow
            0.8,   # left_shoulder: big forward swing
            0.0,   # left_elbow
            0.0    # spine
        ]
        cycle.append(action)
    
    # Left leg MAXIMUM forward
    for _ in range(20):
        action = [
            -1.0,  # right_hip: MAX back
            1.0,   # right_knee: MAX extend
            1.0,   # left_hip: MAX forward
            -0.8,  # left_knee: big bend
            0.8,   # right_shoulder: big forward swing
            0.0,   # right_elbow
            -0.8,  # left_shoulder: big back swing
            0.0,   # left_elbow
            0.0    # spine
        ]
        cycle.append(action)
    
    return cycle


def test_manual_walking(env, num_cycles=10, pattern='simple'):
    """
    Test manual walking with predefined torque sequences
    
    Args:
        env: HumanoidWalkEnv instance
        num_cycles: Number of walking cycles to perform
        pattern: 'simple', 'complex', or 'extreme'
    """
    
    print(f"\n{'='*60}")
    print(f"Testing Manual Walking - Pattern: {pattern}")
    print(f"{'='*60}\n")
    
    # Get walking pattern
    if pattern == 'simple':
        cycle = simple_walking_pattern()
    elif pattern == 'extreme':
        cycle = extreme_walking_pattern()
    else:
        cycle = walking_gait_sequence()
    
    print(f"Walking cycle length: {len(cycle)} steps")
    print(f"Total steps planned: {len(cycle) * num_cycles}")
    print(f"\nStarting in 2 seconds...\n")
    time.sleep(2)
    
    # Reset environment
    obs, info = env.reset()
    
    total_reward = 0
    step_count = 0
    start_pos = obs[0]  # initial x position
    
    # Execute walking cycles
    for cycle_num in range(num_cycles):
        print(f"\nCycle {cycle_num + 1}/{num_cycles}")
        
        for step_in_cycle, action in enumerate(cycle):
            action_array = np.array(action, dtype=np.float32)
            
            # Take step in environment (YOUR env expects 9 actions)
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
            
            # Small delay for visualization if rendering
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
    print(f"Average reward/step: {total_reward/step_count:.4f}")
    print(f"Final height: {obs[2]:.3f}m")
    print(f"{'='*60}\n")
    
    return total_distance, total_reward


def test_single_joint(env, joint_idx, torque_value, num_steps=100):
    """
    Test a single joint to see what it does
    
    Args:
        env: HumanoidWalkEnv
        joint_idx: Which joint to test (0-8)
        torque_value: Torque to apply (-1.0 to 1.0)
        num_steps: How many steps to apply
    """
    joint_names = [
        "right_hip_y", "right_knee", "left_hip_y", "left_knee",
        "right_shoulder1", "right_elbow", "left_shoulder1", "left_elbow", "abdomen_y"
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing Joint: {joint_names[joint_idx]} (index {joint_idx})")
    print(f"Torque: {torque_value}")
    print(f"{'='*60}\n")
    
    obs, info = env.reset()
    
    action = np.zeros(9, dtype=np.float32)
    action[joint_idx] = torque_value
    
    for step in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step:3d} | Height: {obs[2]:.3f}m | X: {obs[0]:.3f}m")
        
        if terminated or truncated:
            print(f"Terminated at step {step}")
            break
        
        if env.render_mode == "human":
            time.sleep(0.02)
    
    print(f"\nFinal position: X={obs[0]:.3f}m, Z={obs[2]:.3f}m\n")


def diagnostic_test_all_joints(env):
    """
    Test each joint individually to understand what each does
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC: Testing Each Joint Individually")
    print("="*60 + "\n")
    
    joint_names = [
        "right_hip_y", "right_knee", "left_hip_y", "left_knee",
        "right_shoulder1", "right_elbow", "left_shoulder1", "left_elbow", "abdomen_y"
    ]
    
    for joint_idx in range(9):
        print(f"\n>>> Testing {joint_names[joint_idx]} with POSITIVE torque (0.8)...")
        test_single_joint(env, joint_idx, torque_value=0.8, num_steps=50)
        time.sleep(0.5)
        
        print(f"\n>>> Testing {joint_names[joint_idx]} with NEGATIVE torque (-0.8)...")
        test_single_joint(env, joint_idx, torque_value=-0.8, num_steps=50)
        time.sleep(0.5)


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    
    print("\n" + "="*70)
    print("        MANUAL WALKING TEST SUITE")
    print("="*70)
    
    # Create environment with rendering to SEE what happens
    print("\n🚀 Initializing environment with rendering...")
    env = HumanoidWalkEnv(render_mode="human")  # Change to None for faster testing
    
    print("\n📋 Choose test mode:")
    print("   1. Diagnostic - Test each joint individually (RECOMMENDED FIRST)")
    print("   2. Simple walking pattern")
    print("   3. Complex walking pattern")
    print("   4. EXTREME walking (exaggerated movements)")
    
    # UNCOMMENT THE TEST YOU WANT TO RUN:
    
    # Option 1: Test individual joints (DO THIS FIRST!)
    diagnostic_test_all_joints(env)
    
    # Option 2: Test simple walking pattern
    # print("\n🚶 Testing SIMPLE walking pattern...")
    # test_manual_walking(env, num_cycles=5, pattern='simple')
    
    # Option 3: Test complex walking pattern
    print("\n🚶 Testing COMPLEX walking pattern...")
    # test_manual_walking(env, num_cycles=5, pattern='complex')
    
    # Option 4: Test EXTREME walking (to see if anything moves)
    # print("\n🚶 Testing EXTREME walking pattern...")
    # test_manual_walking(env, num_cycles=3, pattern='extreme')
    
    env.close()
    print("\n✅ Test complete!")