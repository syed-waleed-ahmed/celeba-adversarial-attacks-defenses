import subprocess
import sys

# Simple runner so Matthijs can reproduce baseline quickly
# Usage:
#   python experiments/run_baseline.py
def main():
    cmd = [
        sys.executable, "-m", "training.train_baseline",
        "--attr", "Smiling",
        "--epochs", "3",
        "--backbone", "resnet18",
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
