# mncr - multi-node checkpoint/restore
#
# Everything here runs without a GPU. The parts that need one are named
# explicitly and say so when they cannot run.

PY      ?= python3
PKGS     = mncr audit torchckpt agent coord imagestore k8s ncclx verify tests

.PHONY: help
help:
	@echo "make check        compile every module and validate every manifest"
	@echo "make test         unit and protocol tests (no GPU needed)"
	@echo "make chaos        fault injection matrix"
	@echo "make scale        scale ladder"
	@echo "make verify       check + test + chaos"
	@echo "make auditor      build the LD_PRELOAD allocation auditor (Linux)"
	@echo "make p0           run the P0 audit against this host"
	@echo "make clean"

.PHONY: check
check:
	@$(PY) -m compileall -q $(PKGS) && echo "compile: ok"
	@$(PY) -c "import yaml, glob, sys; \
	  files = sorted(glob.glob('k8s/crds/*.yaml') + glob.glob('k8s/manifests/*.yaml')); \
	  n = sum(len([d for d in yaml.safe_load_all(open(f)) if d]) for f in files); \
	  print(f'manifests: ok ({n} documents in {len(files)} files)')"
	@$(MAKE) -s -C audit/interpose check && echo "interposer: ok"
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

.PHONY: auditor
auditor:
	$(MAKE) -C audit/interpose

.PHONY: p0
p0:
	@mkdir -p /tmp/mncr-p0
	./audit/fleet_audit.sh > /tmp/mncr-p0/$$(hostname).json
	$(PY) audit/report.py --fleet '/tmp/mncr-p0/*.json' --calls '/tmp/mncr-audit*.jsonl*' || true

.PHONY: clean
clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	$(MAKE) -C audit/interpose clean
