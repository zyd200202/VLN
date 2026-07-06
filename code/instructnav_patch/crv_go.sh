#!/bin/bash
# One-shot preflight + launch for CRV. Run this once the terminal backend is back.
#   bash crv_go.sh
# It (1) syntax-checks the edited objnav_benchmark.py, (2) verifies the CRV assumption
# that make_plan does `self.chainon_answer = self.query_chainon(...)` (so flipping the
# returned dict's Flag actually drives found_goal), and (3) only if BOTH pass, launches
# run_crv.sh in the background (chained to wait for any current GPU job to exit).
set +u
cd /root/autodl-tmp/InstructNav
PY=/root/autodl-tmp/envs/habitat/bin/python3.9

echo "=== 1) py_compile objnav_benchmark.py ==="
if $PY -m py_compile objnav_benchmark.py; then echo "  COMPILE OK"; else echo "  COMPILE FAIL -> fix syntax before running"; exit 1; fi

echo "=== 2) verify  self.chainon_answer <- self.query_chainon(...) ==="
$PY - <<'PYEOF'
import marshal, types, dis, sys
def load(p):
    f=open(p,'rb'); f.read(16); return marshal.load(f)
def find(c,n):
    for k in c.co_consts:
        if isinstance(k,types.CodeType):
            if k.co_name==n: return k
            r=find(k,n)
            if r: return r
co=load("objnav_agent.pyc"); mp=find(co,"make_plan")
ins=list(dis.get_instructions(mp))
ok=False
for i,x in enumerate(ins):
    if x.argrepr=="query_chainon" and x.opname in ("LOAD_METHOD","LOAD_ATTR"):
        for y in ins[i:i+10]:
            if y.opname=="STORE_ATTR" and y.argrepr=="chainon_answer":
                ok=True; break
print("  VERIFY:", "PASS  (CRV query_chainon-flip is valid)" if ok
      else "FAIL  (return not stored as chainon_answer -> switch CRV to a make_plan post-hoc found_goal patch)")
sys.exit(0 if ok else 2)
PYEOF
VOK=$?
if [ "$VOK" -ne 0 ]; then
  echo "  -> NOT launching. CRV needs the make_plan-patch fallback; tell me and I'll switch it."
  exit 2
fi

echo "=== 3) launch run_crv.sh (background, GPU-chained) ==="
mkdir -p /root/autodl-tmp/DRPhysNav/runs/crv
setsid bash run_crv.sh > /root/autodl-tmp/DRPhysNav/runs/crv/launch.out 2>&1 < /dev/null & disown
echo "  launched run_crv.sh (pid $!). tail -f /root/autodl-tmp/DRPhysNav/runs/crv/crv.log"
