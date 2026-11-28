"""Run the AAB-L pipeline: the step files in this folder executed in order in ONE shared namespace, in order.

    python src/run.py                  # everything (00 ... 70)
    python src/run.py --until 30       # runtime, config, data, modules, checks (no training)
    python src/run.py --skip 41 70     # skip the training grid (results from weights already in the output folder) and the demo
    python src/run.py --list

A step number is the file prefix (00, 01, 10, ...). The files are not importable modules: each relies on the globals created
by the files before it. Skip only steps that produce results (41 = training grid, 5x, 60, 70);
the other files define functions and objects that later steps need.
"""
import argparse, glob, os, time

SRC = os.path.dirname(os.path.abspath(__file__))
STEPS = sorted(glob.glob(os.path.join(SRC, "[0-9][0-9]_*.py")))


def step_no(p):
    return int(os.path.basename(p)[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--until", type=int, default=99, help="last step to run (inclusive)")
    ap.add_argument("--skip", type=int, nargs="*", default=[], help="steps to leave out")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()
    if a.list:
        print("\n".join(os.path.basename(p) for p in STEPS))
        return
    todo = [p for p in STEPS if step_no(p) <= a.until and step_no(p) not in a.skip]
    ns = {"__name__": "__main__", "__builtins__": __builtins__}
    for p in todo:
        t0 = time.time()
        print(f"\n===== {os.path.basename(p)} =====", flush=True)
        with open(p, encoding="utf-8") as f:
            exec(compile(f.read(), p, "exec"), ns)
        print(f"===== {os.path.basename(p)} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
