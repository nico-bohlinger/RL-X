import argparse

import wandb


parser = argparse.ArgumentParser()
parser.add_argument("leader_run_id")
parser.add_argument("--leader-label", default="G1 current-best FastMPO")
args = parser.parse_args()

entity = "nico-bohlinger"
project = "custom_mujoco_robot_locomotion"
dashboard_tag = "g1-best-fastmpo-vs-baselines-dashboard"
leader_tag = "g1-current-best-fastmpo"
baselines = {
    "s14111zd": ("G1 PPO baseline [s14111zd]", "g1-baseline-ppo"),
    "rcul98qz": ("G1 FastSAC baseline [rcul98qz]", "g1-baseline-fastsac"),
    "2xrt4qcy": ("G1 FastTD3 baseline [2xrt4qcy]", "g1-baseline-fasttd3"),
    "k9plqga1": ("G1 FlashSAC baseline [k9plqga1]", "g1-baseline-flashsac"),
    "r24wfzwv": ("G1 RePPO baseline [r24wfzwv]", "g1-baseline-reppo"),
}
selected_run_ids = set(baselines) | {args.leader_run_id}
api = wandb.Api(timeout=60)

for run in api.runs(f"{entity}/{project}", filters={"tags": {"$in": [dashboard_tag]}}):
    if run.id not in selected_run_ids:
        run.tags = tuple(tag for tag in run.tags if tag not in {dashboard_tag, leader_tag})
        run.update()

for run_id, (display_name, role_tag) in baselines.items():
    run = api.run(f"{entity}/{project}/{run_id}")
    run.name = display_name
    run.tags = tuple(sorted(set(run.tags) | {dashboard_tag, role_tag}))
    run.update()

leader = api.run(f"{entity}/{project}/{args.leader_run_id}")
leader.name = f"{args.leader_label} [{args.leader_run_id}]"
leader.tags = tuple(sorted(set(leader.tags) | {dashboard_tag, leader_tag}))
leader.update()

tagged = list(api.runs(f"{entity}/{project}", filters={"tags": {"$in": [dashboard_tag]}}))
tagged_run_ids = {run.id for run in tagged}
if tagged_run_ids != selected_run_ids:
    raise RuntimeError(f"Dashboard run mismatch: expected {sorted(selected_run_ids)}, found {sorted(tagged_run_ids)}")

print("https://wandb.ai/nico-bohlinger/custom_mujoco_robot_locomotion?nw=6fklp0vmaus")
print("Selected runs:", ", ".join(sorted(tagged_run_ids)))
