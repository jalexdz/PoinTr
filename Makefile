SHELL := /bin/bash
COMPOSE := docker compose -f docker/docker-compose.yaml

.PHONY: build shell up down

build:
	$(COMPOSE) build

shell:
	HOST_DATA_ROOT=$${HOST_DATA_ROOT:-$$HOME/datasets} $(COMPOSE) run --rm adapointr bash

up:
	HOST_DATA_ROOT=$${HOST_DATA_ROOT:-$$HOME/datasets} $(COMPOSE) up -d

down:
	$(COMPOSE) down
