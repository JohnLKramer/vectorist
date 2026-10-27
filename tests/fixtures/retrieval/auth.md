# Authentication

## Tokens

Clients present a signed bearer token that expires after one hour. Expired tokens are rejected with a 401 response.

## Refresh

A long lived refresh credential can be exchanged for a new bearer token without asking the user to sign in again.

## Passwords

Passwords are hashed with a slow salted algorithm and are never stored in plain text.
