SHELL := /bin/bash
BUILD_DIR    := build
EXECUTABLE   := scoop
VULKAN_SETUP := /home/lkukhale/vulkan/1.4.321.1/setup-env.sh
ENSURE_ENV   := scripts/vulkan_env.sh
.SHELLFLAGS := -euo pipefail -c
.ONESHELL:


ensure-vulkan:
	@copy_clip() {
			if command -v wl-copy >/dev/null 2>&1; then
				printf "%s" "$$1" | wl-copy
				echo "[env] Copied to clipboard via wl-copy."
			elif command -v xclip >/dev/null 2>&1; then
				printf "%s" "$$1" | xclip -selection clipboard
				echo "[env] Copied to clipboard via xclip."
			elif command -v xsel >/dev/null 2>&1; then
				printf "%s" "$$1" | xsel --clipboard --input
				echo "[env] Copied to clipboard via xsel."
			elif command -v pbcopy >/dev/null 2>&1; then
				printf "%s" "$$1" | pbcopy
				echo "[env] Copied to clipboard via pbcopy."
			else
				echo "[env] No clipboard tool found (install wl-clipboard, xclip, or xsel)."
			fi
	}
	@if command -v glslc >/dev/null 2>&1 || [[ -n "$${VULKAN_SDK-}" ]]; then
	  echo "[env] Vulkan environment present."
	else
	  echo "[env] Vulkan SDK not found (no glslc and no \$${VULKAN_SDK})."
	  echo

	  goals="$(MAKECMDGOALS)"; [[ -n "$$goals" ]] || goals="$@"
	  quoted_goals=$$(printf '%q ' $$goals)
	  cmd="source $(ENSURE_ENV) "$(VULKAN_SETUP)" && $(MAKE) $$quoted_goals"

	  echo "Warrning: by pressing Enter you'll only set SDK for make's shell and will have to re source on other runs"
	  echo "If you want to set SDK for this sesion press Ctrl+C and then just press Ctrl+Shift+V to run the comman manually"
	  echo "Ready to run in your terminal:"
	  echo "    $$cmd"
	  copy_clip "$$cmd"
	  echo
	  read -r -p "Press Enter to run it automatically (Ctrl+C to abort)..." _
	  exec bash -lc "$$cmd"
	fi

all: ensure-vulkan
	@mkdir -p "$(BUILD_DIR)"
	@cmake -S . -B "$(BUILD_DIR)" && cmake --build "$(BUILD_DIR)"

clean:
	@cmake --build "$(BUILD_DIR)" --target clean 2>/dev/null || true

fclean:
	@rm -rf "$(BUILD_DIR)"

re: fclean all

run: all
	@./$(BUILD_DIR)/$(EXECUTABLE)

rerun: re
	@./$(BUILD_DIR)/$(EXECUTABLE)

.PHONY: all clean fclean re run rerun