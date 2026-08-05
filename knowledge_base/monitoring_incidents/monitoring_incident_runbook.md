# Monitoring Incident Runbook

Use this runbook when monitoring capabilities are degraded or unavailable.

## Step 1: determine scope

Identify whether the issue affects:
- Grafana dashboards;
- metric collection;
- Alertmanager notifications;
- one environment or multiple environments.

Check the monitoring service status page and recent deployment activity.

## Step 2: preserve visibility

If dashboards are unavailable but metrics are still being collected, avoid unnecessary changes to collectors.

If alert delivery is unavailable, notify the operations channel that automated alerts may not be reliable and assign manual observation where required.

## Step 3: escalate

Escalate as Severity 1 when monitoring loss creates a material risk of missing a major production outage across multiple services.

A limited dashboard issue is not automatically Severity 1.

## Step 4: record the incident

Record the start time, affected components, actions taken, and recovery time in the incident system.
