/**
 * Entry point for the benchmark.  The app hands this command to whoever is
 * running the demo, so it has to survive landing on a machine with an older
 * Node than this repo develops against: Node only strips TypeScript types by
 * default from 22.18 / 23.6 / 24 on, and without stripping it cannot load
 * scripts/bench.ts at all — not even far enough to print a useful error.
 *
 * So this wrapper is plain JS, and re-executes itself with the flag when the
 * running Node has stripping available but off (22.6 through 22.17).
 */
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

if (process.features.typescript) {
  await import('./bench.ts');
} else {
  const entry = fileURLToPath(new URL('./bench.ts', import.meta.url));
  const result = spawnSync(
    process.execPath,
    [
      '--experimental-strip-types',
      '--disable-warning=ExperimentalWarning',
      entry,
      ...process.argv.slice(2),
    ],
    { stdio: 'inherit' },
  );
  // exit code 9 is node's "bad option": this Node predates type stripping
  if (result.error || result.status === 9) {
    console.error(
      `bench: Node ${process.versions.node} cannot run this project's TypeScript sources.\n` +
        '  Node 22.6+ can with --experimental-strip-types; 22.18, 23.6 and 24+ do it by default.',
    );
    process.exit(1);
  }
  process.exit(result.status ?? 1);
}
