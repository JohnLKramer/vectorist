# Structured Logging

## Log Format

Services emit one JSON object per line so that log aggregators can parse fields without regular expressions.

## Levels

Use debug for developer detail, info for lifecycle events, warn for recoverable problems, and error for failures that need a human to look.

## Correlation

Every request carries a trace identifier that is copied into each log line it produces.
