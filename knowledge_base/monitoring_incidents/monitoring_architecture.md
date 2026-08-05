# Monitoring Architecture Overview

Northstar's service monitoring stack has three major functions: metric collection, visualization, and alert delivery.

## Metrics

Prometheus-compatible collectors gather infrastructure and application metrics from approved environments.

## Dashboards

Grafana is used for dashboards and operational visualization. A Grafana dashboard being unavailable does not necessarily mean metric collection has stopped.

## Alerts

Alertmanager receives alert events and routes notifications to the appropriate operational channels.

## Important distinction

The phrase "monitoring is down" can describe different failures:
- dashboards unavailable;
- metrics not being collected;
- alerts not being delivered;
- only one monitored application failing.

Engineers should identify which monitoring function is affected before declaring a complete monitoring outage.
