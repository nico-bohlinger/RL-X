# Online OmniRetarget++ research handover

The complete machine-transfer handover, experiment history, immutable evidence,
asset inventory and recovery instructions are in the private
[OmniRetarget++ research repository](https://github.com/nico-bohlinger/omni-retarget-plus-plus/blob/main/HANDOVER.md).

This branch preserves the online integration working tree as it stood on
22 September 2026. It is a research snapshot, not a new qualification run or a
merge into RL-X master. No PPO or runner modification is introduced here.

The standalone research benchmarks qualified shared, grouped-mixed and
independently sampled clip assignments for object-only, robot-only and combined
variation. In the RL-X training integration, only robot-only variation with the
original scanned OMOMO object has passed the full interface and learning smoke
checks. Scanned-mesh object-size variation remains unresolved. Box-mode code
must not be represented as support for the scanned object.

The 2B-transition robot-only training run completed, but its tracking evaluation
remained weak. The reference videos are not learned-policy videos. Final policy
capture failed during report writing after saving rollout arrays, and the
report-recovery scripts remain unsubmitted. Exact run identities, hashes,
checkpoint locations, metrics and the remaining work are in the handover.

The current source includes later interface and style fixes. Historical R6
qualification applies to its frozen source archive, not automatically to every
later edit in this branch. Revalidate the current revision before new training.
