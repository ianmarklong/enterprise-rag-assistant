# Container Platform Access

Northstar operates shared container environments for development, staging, and production workloads.

## Access levels

Employees may receive one of three roles:
- Viewer: inspect workloads and logs;
- Developer: deploy to approved development namespaces;
- Operator: perform approved operational actions.

Production Operator access is restricted to authorized operations staff.

## Joining the platform

Employees who need access should:
1. complete the Container Platform Fundamentals training;
2. join the relevant engineering team group;
3. submit an access request through the Access Management Portal;
4. select the environment and required role;
5. obtain manager approval.

Production access also requires approval from the platform owner.

## Local Docker versus shared platform

Docker Desktop is used for local development. It is not the same as the shared container platform and does not provide production access.

## Authentication

The shared platform uses corporate single sign-on for normal user access. Privileged operational actions may require a separate elevated session.
