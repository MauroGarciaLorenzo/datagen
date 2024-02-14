import os
import subprocess


subprocess.run(["pip", "install", "scipy"], check=True)

subprocess.run(["module", "load", "COMPSs/3.3"], check=True)

current_directory = os.getcwd()
print(current_directory)

pythonpath = f"{current_directory}/..:{current_directory}:{current_directory}/../.."
os.environ["PYTHONPATH"] = pythonpath

num_nodes = 1

while num_nodes <= 8:
    subprocess.run([
        "enqueue_compss",
        f"--pythonpath={pythonpath}",
        f"--job_execution_dir={current_directory}/..",
        f"--num_nodes={num_nodes}",
        "--worker_working_dir=local_disk",
        "--master_working_dir=local_disk",
        "--lang=python",
        "--exec_time=400",
        "--agents",
        "--tracing",
        "node_scalability_test.py",
        f"{current_directory}/../results/node_scalability{num_nodes}"
    ], check=True)

    num_nodes *= 2
