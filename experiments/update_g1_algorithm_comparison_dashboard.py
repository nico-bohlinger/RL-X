import argparse

import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--spo")
parser.add_argument("--fpo")
parser.add_argument("--dppo")
parser.add_argument("--dime")
parser.add_argument(
    "--fastmpo",
    action="append",
    default=["6aufiuih", "edo7pc2n"],
)
parser.add_argument("--create-workspace", action="store_true")
args = parser.parse_args()

entity = "nico-bohlinger"
project = "custom_mujoco_robot_locomotion"
dashboard_tag = "g1-algorithm-comparison-dashboard"
run_ids = {
    "s14111zd",
    "rcul98qz",
    "2xrt4qcy",
    "k9plqga1",
    "r24wfzwv",
    *args.fastmpo,
    *(
        run_id
        for run_id in [args.spo, args.fpo, args.dppo, args.dime]
        if run_id
    ),
}
api = wandb.Api(timeout=60)

for run in api.runs(
    f"{entity}/{project}",
    filters={"tags": {"$in": [dashboard_tag]}},
):
    if run.id not in run_ids:
        run.tags = tuple(
            tag for tag in run.tags if tag != dashboard_tag
        )
        run.update()

for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    run.tags = tuple(sorted(set(run.tags) | {dashboard_tag}))
    run.update()

tagged_run_ids = {
    run.id
    for run in api.runs(
        f"{entity}/{project}",
        filters={"tags": {"$in": [dashboard_tag]}},
    )
}
if tagged_run_ids != run_ids:
    raise RuntimeError(
        "Dashboard run mismatch: "
        f"expected {sorted(run_ids)}, found {sorted(tagged_run_ids)}"
    )

if args.create_workspace:
    from wandb_workspaces import workspaces as ws
    from wandb_workspaces.reports.v2 import LinePlot

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name="G1 Locomotion - Algorithms vs Best FastMPO",
        sections=[
            ws.Section(
                name="Curriculum outcome",
                is_open=True,
                panels=[
                    LinePlot(
                        title="Curriculum coefficient vs global step",
                        x="global_step",
                        y=["env_curriculum/coefficient"],
                        title_x="Global step",
                        title_y="Curriculum coefficient",
                        smoothing_type="none",
                        max_runs_to_show=16,
                    ),
                    LinePlot(
                        title="Curriculum coefficient vs wall-clock time",
                        x="_runtime",
                        y=["env_curriculum/coefficient"],
                        title_x="Runtime (seconds)",
                        title_y="Curriculum coefficient",
                        smoothing_type="none",
                        max_runs_to_show=16,
                    ),
                ],
            ),
        ],
        settings=ws.WorkspaceSettings(
            x_axis="global_step",
            smoothing_type="none",
            max_runs=16,
        ),
        runset_settings=ws.RunsetSettings(
            filters=[ws.Tags().isin([dashboard_tag])],
            pinned_columns=[
                "run:displayName",
                "summary:env_curriculum/coefficient",
                "summary:global_step",
            ],
        ),
    )
    saved_workspace = workspace.save_as_new_view()
    print(saved_workspace.url)

print("Selected runs:", ", ".join(sorted(tagged_run_ids)))
