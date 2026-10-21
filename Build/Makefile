APP_NAME = Reelpack
SWIFT_PRODUCT = Reelpack
CLI_PRODUCT = reelpack-cli
VERSION ?= $(shell cat VERSION)
ARCH ?= $(shell uname -m)
PYTHON ?= python3
OUTPUT_DIR ?= dist/$(APP_NAME)-$(VERSION)-$(ARCH)
SIGN_IDENTITY ?= -
export MACOSX_DEPLOYMENT_TARGET = 14.0

.PHONY: all build bundle archive
all: build

# Run from the repository root: make -f Build/Makefile ARCH=arm64 bundle
build:
	swift build -c release --arch $(ARCH) --product $(SWIFT_PRODUCT)
	swift build -c release --arch $(ARCH) --product $(CLI_PRODUCT)

# Produces Reelpack.app, a zip, SHA-256 and a size/component manifest.
# Existing artifacts are preserved; choose another OUTPUT_DIR for a rebuild.
bundle:
	$(PYTHON) Build/build_app.py --arch $(ARCH) --version $(VERSION) --output-dir "$(OUTPUT_DIR)" --sign-identity "$(SIGN_IDENTITY)"

archive: bundle
