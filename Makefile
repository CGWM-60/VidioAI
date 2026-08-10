.PHONY: dev test build release deploy smoke create-server destroy-server

dev:
	./scripts/dev.sh

stop:
	./scripts/stop.sh

test:
	cd backend && cargo clippy --locked --all-targets -- -D warnings && cargo test --locked
	cd frontend && npm run lint && npm run build
	python -m pytest -q worker

build:
	BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker compose --progress plain build

release:
	./deploy/scripts/build-release.sh "$(VERSION)"

deploy:
	./deploy/scripts/deploy.sh "$(VERSION)"

smoke:
	./deploy/scripts/smoke-test.sh "$(URL)"

create-server:
	./deploy/scaleway/create-server.sh

destroy-server:
	./deploy/scaleway/destroy-server.sh "--confirm-destroy=$(VIDIOAI_SERVER_NAME)"
