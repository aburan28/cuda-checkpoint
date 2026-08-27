# mncr - multi-node checkpoint/restore
#
# Everything here runs without a GPU. The parts that need one are named
# explicitly and say so when they cannot run.

PY      ?= python3
PKGS     = mncr audit torchckpt agent coord imagestore k8s ncclx verify tests

.PHONY: help
help:
	@echo "no hardware needed"
	@echo "  make check       compile every module and validate every manifest"
	@echo "  make test        unit and protocol tests"
	@echo "  make chaos       fault injection matrix"
	@echo "  make scale       scale ladder"
	@echo "  make verify      check + test + chaos"
	@echo "  make interposer  build and test the allocation auditor"
	@echo ""
	@echo "on a GPU node"
	@echo "  make preflight   can this node participate? (real backends)"
	@echo "  make smoke       driver, criu, agent and a full epoch, for real"
	@echo "  make p0          fleet audit and allocation report"
	@echo "  make auditor     build the LD_PRELOAD allocation auditor"
	@echo ""
	@echo "packaging"
	@echo "  make images      build the agent and control container images"
	@echo "  make clean"

.PHONY: check
check:
	@$(PY) -m compileall -q $(PKGS) mncrctl && echo "compile: ok"
	@$(PY) -c "import yaml, glob, sys; \
	  files = sorted(glob.glob('k8s/crds/*.yaml') + glob.glob('k8s/manifests/*.yaml')); \
	  n = sum(len([d for d in yaml.safe_load_all(open(f)) if d]) for f in files); \
	  print(f'manifests: ok ({n} documents in {len(files)} files)')"
	@$(MAKE) -s -C audit/interpose test
	@shellcheck audit/fleet_audit.sh && echo "shell: ok"

.PHONY: test
test:
	MNCR_LOG_LEVEL=error $(PY) -m unittest discover -s tests -t . -v

.PHONY: chaos
chaos:
	MNCR_LOG_LEVEL=error $(PY) -m verify.chaos

.PHONY: scale
scale:
	MNCR_LOG_LEVEL=error $(PY) -m verify.scale --repeats 3 --max-nodes 8 --max-ranks 8

.PHONY: verify
verify: check test chaos

.PHONY: preflight
preflight:
	$(PY) -m agent.preflight

.PHONY: smoke
smoke:
	$(PY) -m verify.smoke

.PHONY: images
images:
	./prepare-image-context.sh
	docker build --target control -t mncr-control:$(shell $(PY) -c "import mncr; print(mncr.__version__)") .
	docker build --target agent   -t mncr-agent:$(shell $(PY) -c "import mncr; print(mncr.__version__)") .

.PHONY: auditor interposer
auditor:
	$(MAKE) -C audit/interpose

# Builds the interposer and runs it against a stand-in driver. On Linux that is
# the production LD_PRELOAD path; on macOS it links directly, which covers the
# logic but not the preload mechanism.
interposer:
	$(MAKE) -C audit/interpose test

.PHONY: p0
p0:
	@mkdir -p /tmp/mncr-p0
	./audit/fleet_audit.sh > /tmp/mncr-p0/$$(hostname).json
	$(PY) audit/report.py --fleet '/tmp/mncr-p0/*.json' --calls '/tmp/mncr-audit*.jsonl*' || true

.PHONY: clean
clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf vendor-bin
	$(MAKE) -C audit/interpose clean
