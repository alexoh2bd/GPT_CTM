import subprocess
import time
import os

def get_processes_using_metal():
    """Find processes that might be using Metal/MPS"""
    try:
        # Get processes with high memory usage (potential GPU users)
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')[1:]  # Skip header
        
        gpu_processes = []
        for line in lines:
            if line.strip():
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    mem_percent = float(parts[3])
                    command = parts[10]
                    
                    # Look for potential MPS users
                    if (mem_percent > 5.0 or 
                        'python' in command.lower() or 
                        'metal' in command.lower() or
                        'tensorflow' in command.lower() or
                        'pytorch' in command.lower()):
                        gpu_processes.append({
                            'pid': parts[1],
                            'user': parts[0],
                            'cpu': parts[2],
                            'mem': parts[3],
                            'command': command[:50]
                        })
        
        return gpu_processes[:10]  # Top 10
    except:
        return []

def get_gpu_power():
    """Get GPU power consumption"""
    try:
        result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'gpu_power', '-n', '1'], capture_output=True, text=True)
        # Parse power metrics (simplified)
        for line in result.stdout.split('\n'):
            if 'GPU Power' in line:
                return line.strip()
        return "GPU Power: N/A"
    except:
        return "GPU Power: N/A"

def main():
    while True:
        os.system('clear')
        print("=== MPS/Metal Process Monitor ===")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print()
        
        # Show GPU power
        print(get_gpu_power())
        print()
        
        # Show potential MPS processes
        processes = get_processes_using_metal()
        print(f"{'PID':<8} {'USER':<10} {'CPU%':<6} {'MEM%':<6} {'COMMAND':<50}")
        print("-" * 80)
        
        for proc in processes:
            print(f"{proc['pid']:<8} {proc['user']:<10} {proc['cpu']:<6} "
                    f"{proc['mem']:<6} {proc['command']:<50}")
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")