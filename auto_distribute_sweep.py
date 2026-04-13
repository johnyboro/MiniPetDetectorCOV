import os
import multiprocessing as mp
import torch
import wandb

def run_agent_on_gpu(gpu_id: int, sweep_id: str, function, count: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WANDB_AGENT_LABEL"] = f"gpu-{gpu_id}"
    wandb.agent(sweep_id=sweep_id, function=function, count=count)

def spawn_multiple_agents(sweep_id: str, function, count: int):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected, running one CPU agent.")
        wandb.agent(sweep_id=sweep_id, function=function, count=count)
    else:
        procs = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=run_agent_on_gpu, args=(gpu_id, sweep_id, function, count))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
