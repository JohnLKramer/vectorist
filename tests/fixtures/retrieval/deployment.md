# Deployment

## Rollout

New versions are released to a small share of servers first. If error rates stay flat, the release continues in stages until every server runs it.

## Rollback

A failed release is reverted by redeploying the previous container image, which takes under two minutes.

## Configuration

Environment specific settings are injected at startup and are never baked into the image.
