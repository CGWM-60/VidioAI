.PHONY: dev test build release deploy smoke create-server destroy-server

dev:
	./scripts/dev.sh

stop:
	./scripts/stop.sh

test:
	cd backend && cargo clippy --locked --all-targets -- -D warnings && cargo test --locked
	cd frontend && npm run lint && npm run build
	python -m pytest -q worker
	bash deploy/tests/test-s3-paths.sh
	@if [ "$${VIDIOAI_RUN_COMPOSE_TESTS:-true}" = "true" ]; then bash deploy/tests/test-compose-orchestration.sh; fi

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
