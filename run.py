import sys

if len(sys.argv) < 2:
    print("Usage: run.py [experiment name]")

# Run experiment
experiment_name = sys.argv[1]
print("Running experiment", experiment_name)

__import__('experiments.' + experiment_name)