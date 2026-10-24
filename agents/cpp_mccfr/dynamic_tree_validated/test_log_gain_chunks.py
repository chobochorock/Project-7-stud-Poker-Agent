"""Small controller checks; no large training.

Usage: python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_log_gain_chunks -v
Checks original preservation, chunk parity, stop-file save and explicit resume.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import log_gain, log_gain_chunks as chunks, run, test_log_gain
from .test_schedule import solver_payload


class ChunkTest(unittest.TestCase):
    def test_chunk_parity_and_handoff(self):
        with tempfile.TemporaryDirectory(dir=run.DATA, prefix="test_chunks_") as name:
            root = Path(name).resolve()
            self.assertTrue(root.is_relative_to(run.DATA.resolve()))
            source, full, split = (root / s for s in ("source", "full", "split"))
            options = test_log_gain.LogGainTest().options
            self.assertEqual(log_gain.train(options(source, 32)), 0)
            digest = run.legacy.sha256(source / "checkpoint.bin")
            chunks.fork(source, full)
            chunks.fork(source, split)
            self.assertEqual(log_gain.train(options(full, 64, ("--resume",))), 0)
            args = chunks.parser().parse_args(
                [
                    "train",
                    "--run",
                    str(split),
                    "--hands",
                    "64",
                    "--chunk-hands",
                    "16",
                    "--checkpoint-every",
                    "16",
                    "--reserve-gib",
                    "0",
                    "--min-free-gib",
                    "0",
                ]
            )
            self.assertEqual(chunks.train(args), 0)
            self.assertEqual(solver_payload(full), solver_payload(split))
            self.assertTrue((split / "review_48/summary.json").is_file())
            self.assertEqual(digest, run.legacy.sha256(source / "checkpoint.bin"))
            command = [
                sys.executable,
                "-m",
                "agents.cpp_mccfr.dynamic_tree_validated.log_gain_chunks",
                "train",
                "--run",
                str(split),
                "--hands",
                "20000",
                "--chunk-hands",
                "20000",
                "--checkpoint-every",
                "16",
                "--reserve-gib",
                "0",
                "--min-free-gib",
                "0",
            ]
            lines = []
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            ) as process:
                for line in process.stdout:
                    lines.append(line)
                    if line.startswith("CONFIG "):
                        (split / "STOP_REQUESTED").touch()
                self.assertEqual(process.wait(timeout=30), 2, "".join(lines))
            saved = chunks.checkpoint_header(
                split, log_gain.manifest_at(split)["identity"]
            )
            self.assertGreater(saved, 64)
            self.assertLess(saved, 20000)
            self.assertIn("PAUSED reason=user_stop", "".join(lines))
            resumed = options(full, saved, ("--resume",))
            self.assertEqual(log_gain.train(resumed), 0)
            self.assertEqual(solver_payload(full), solver_payload(split))
            args.hands = saved + 1
            self.assertEqual(chunks.train(args), 0)
            self.assertFalse((split / "STOP_REQUESTED").exists())
            self.assertEqual(digest, run.legacy.sha256(source / "checkpoint.bin"))


if __name__ == "__main__":
    unittest.main()
