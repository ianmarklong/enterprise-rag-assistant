# GPU Compute Infrastructure

Northstar provides shared GPU capacity for machine-learning development, model evaluation, and approved inference workloads.

## Development GPU pool

The development pool contains shared NVIDIA GPUs and is intended for experimentation, testing, and short-running jobs.

Jobs are scheduled through the internal compute platform. Employees should not assume a specific GPU model will always be available in the shared development pool.

## High-memory GPU pool

A smaller high-memory pool is reserved for workloads that cannot run efficiently on the standard development pool. Access requires justification because capacity is limited.

## Production inference

Production inference workloads run in managed environments operated by the platform team. Engineers should not deploy production models directly from personal workstations.

## Access

GPU access requires:
- membership in an approved engineering or data team;
- completion of the compute usage orientation;
- an approved compute access request.

## Usage expectations

Users should release unused reservations, define reasonable resource requests, and avoid storing long-term data on local GPU node disks.
