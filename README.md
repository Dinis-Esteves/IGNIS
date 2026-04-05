# IGNIS
IGNIS — Integrated GPU Network for Intensive Scientific Scheduling

## Regenerating Protobuf files

Run this after creating or modifying any `.proto` file:

```bash
PATH=$PATH:$HOME/go/bin protoc \
  --go_out=. --go_opt=paths=source_relative \
  --go-grpc_out=. --go-grpc_opt=paths=source_relative \
  common/proto/*.proto
```
