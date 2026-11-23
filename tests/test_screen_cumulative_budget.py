"""Regression tests for cumulative JSONL screening budgets."""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from screen import main


class CumulativeInputBudgetTests(unittest.TestCase):
    def run_cli(self, payload, *args):
        output, errors = io.StringIO(), io.StringIO()
        status = main(list(args), stdin=io.StringIO(payload), stdout=output, stderr=errors)
        return status, output.getvalue(), errors.getvalue()

    def test_excess_cumulative_input_fails_atomically(self):
        line = '{"text":"hello"}\\n'.replace('\\n', '\n')
        status, output, errors = self.run_cli(line * 2, "--max-input-chars", str(len(line) * 2 - 1))
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("batch exceeds --max-input-chars limit", errors)
        self.assertNotIn("hello", errors)

    def test_exact_budget_succeeds(self):
        line = '{"text":"hello"}\\n'.replace('\\n', '\n')
        status, output, errors = self.run_cli(line, "--max-input-chars", str(len(line)))
        self.assertEqual(status, 0, errors)
        self.assertEqual(len(output.splitlines()), 1)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            main(["--max-input-chars", "0"], stdin=io.StringIO(""))
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
