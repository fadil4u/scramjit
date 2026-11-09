"""The test suite in chunks of files, each in a fresh pytest process, so memory stays bounded (a single process
over every file grows past what a small machine or CI runner has). Extra arguments go to every pytest run.

    python tools/run_tests.py [--step 12] [pytest args...]

Exits 1 if any chunk fails."""

import glob
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main(argv):
    step = 12
    if argv[:1] == ['--step']:
        step, argv = int(argv[1]), argv[2:]
    files = sorted(glob.glob(os.path.join(ROOT, 'tests', 'test_*.py')))
    failed = []
    for i in range(0, len(files), step):
        chunk = files[i:i + step]
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *argv, *chunk], cwd=ROOT)
        names = ' '.join(os.path.basename(f) for f in chunk)
        print(f'== chunk {i // step}: {"ok" if r.returncode in (0, 5) else "FAILED"} in {time.perf_counter() - t0:.0f} s: {names}',
              flush=True)
        if r.returncode not in (0, 5):          # 5: no tests collected
            failed.append(names)
    for names in failed:
        print('FAILED chunk:', names)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
