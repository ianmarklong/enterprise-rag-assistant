# Container Troubleshooting Guide

This guide covers common failures for Northstar containerized applications.

## Container keeps restarting

First inspect:
- container status;
- recent application logs;
- exit code;
- configured health checks;
- memory and CPU limits.

A restart loop may be caused by application crashes, missing configuration, failed dependencies, failed health checks, or resource limits.

## Cannot access the container platform

Confirm that:
- the corporate account is active;
- MFA is working;
- the employee has the correct platform role;
- the requested environment is included in the access grant.

Installing Docker Desktop does not provide access to shared Northstar environments.

## Production incidents

If a production workload failure causes a customer-facing outage, follow the incident management process rather than repeatedly restarting containers without diagnosis.
