# WhisperX — locked uv install + run (defaults adapted for macOS CPU)
#
# Examples:
#   make install
#   make run AUDIO=~/Downloads/recording.wav
#   make run AUDIO=sample.mp3 EXTRA="--language fr"
#   make whisper ARGS="sample.wav --model large-v2 --diarize --hf_token HF_XXX"

VENV        ?= .venv
UV          ?= uv
WHISPERX    := $(VENV)/bin/whisperx

.DEFAULT_GOAL := help

.PHONY: help install test run whisper clean

help: ## Show available targets
	@echo "WhisperX Makefile"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make run AUDIO=path/to/audio.wav"
	@echo "  make whisper ARGS=\"audio.wav --language fr\""

install: ## Install locked dependencies using uv and .python-version
	UV_PROJECT_ENVIRONMENT="$(VENV)" $(UV) sync --locked --extra dev

test: ## Run tests with the locked development dependencies
	UV_PROJECT_ENVIRONMENT="$(VENV)" $(UV) run --locked --extra dev pytest tests/ -q

run: ## Transcribe AUDIO=path [-- EXTRA='optional whisperx flags']
	@test -n "$(AUDIO)" || (echo "Usage: make run AUDIO=path/to/audio.wav [EXTRA='--language fr']"; exit 1)
	@test -x $(WHISPERX) || (echo "Run \`make install\` first."; exit 1)
	$(WHISPERX) "$(AUDIO)" --compute_type int8 --device cpu $(EXTRA)

whisper: ## Full CLI passthrough: make whisper ARGS="file.wav ..."
	@test -n "$(ARGS)" || (echo "Usage: make whisper ARGS=\"path/to/audio.wav [flags...]\""; exit 1)
	@test -x $(WHISPERX) || (echo "Run \`make install\` first."; exit 1)
	$(WHISPERX) $(ARGS)

clean: ## Remove local .venv
	rm -rf $(VENV)
