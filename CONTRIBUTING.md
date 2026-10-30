# Contributing to pyshell

<br />

## Development

Run from the repository root with Make, Docker, and Docker Compose v2.24+:

```bash
make up
```

Cleanup:

```bash
make down
```

## Testing

```bash
make test
```

## Release

```bash
git fetch origin main
git tag vX.Y.Z origin/main
git push origin vX.Y.Z
```
