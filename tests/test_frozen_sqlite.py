"""SQL-derived grades must not depend on the system clock or CPU speed."""
from pathlib import Path
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import apsw
from measurement_db.scripts.build_measurement_tables.frozen_sqlite import FrozenSQLite


class FrozenSQLiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "example.sqlite"
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE values_table(value INTEGER)")
            connection.executemany("INSERT INTO values_table VALUES (?)", [(1,), (2,), (2,)])
        self.original = self.path.read_bytes()
        self.settings = dict(sqlite_version="3.53.2", evaluation_time_utc="2026-10-02T00:00:00Z",
                             vm_steps_per_query=1000)

    def test_sql_clock_is_fixed_without_rewriting_the_query(self):
        with FrozenSQLite(**self.settings) as runtime:
            for _ in range(2):
                with runtime.connection(self.path) as connection:
                    self.assertEqual(connection.execute(
                        "SELECT CURRENT_TIMESTAMP, CURRENT_DATE, CURRENT_TIME, "
                        "date('now'), julianday('now'), strftime('%Y', 'now')").fetchall(),
                        [("2026-10-02 00:00:00", "2026-10-02", "00:00:00", "2026-10-02", 2461315.5, "2026")])
            with runtime.connection(self.path) as connection:
                self.assertEqual(connection.execute("SELECT * FROM values_table").fetchall(), [(1,), (2,), (2,)])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_each_query_gets_a_fresh_instruction_allowance(self):
        query = "WITH RECURSIVE x(n) AS (VALUES(0) UNION ALL SELECT n+1 FROM x WHERE n<1000000) SELECT sum(n) FROM x"
        with FrozenSQLite(**self.settings) as runtime:
            for _ in range(2):
                with runtime.connection(self.path) as connection:
                    with self.assertRaises(apsw.InterruptError):
                        connection.execute(query).fetchall()
            with runtime.connection(self.path) as connection:
                # Elapsed time consumes no instructions and must not change a grade.
                connection.create_scalar_function("delay", lambda value: (time.sleep(0.01), value)[1], 1)
                self.assertEqual(connection.execute("SELECT delay(value) FROM values_table").fetchall(), [(1,), (2,), (2,)])

    def test_localtime_uses_utc_and_restores_the_host_timezone(self):
        try:
            with patch.dict(os.environ, {"TZ": "America/Los_Angeles"}):
                time.tzset()
                with FrozenSQLite(**self.settings) as runtime, runtime.connection(self.path) as connection:
                    self.assertEqual(connection.execute("SELECT datetime('now', 'localtime')").fetchall(),
                                     [("2026-10-02 00:00:00",)])
                self.assertEqual(os.environ["TZ"], "America/Los_Angeles")
        finally:
            time.tzset()

    def test_captured_wal_database_needs_no_journal_or_shared_memory_files(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
        connection.close()
        original = self.path.read_bytes()
        with FrozenSQLite(**self.settings) as runtime, runtime.connection(self.path) as connection:
            self.assertEqual(connection.execute("SELECT sum(value) FROM values_table").fetchall(), [(5,)])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_writes_attach_and_invalid_sql_preserve_original_database(self):
        with FrozenSQLite(**self.settings) as runtime:
            for query in ("DELETE FROM values_table", "CREATE TABLE other(value)",
                          "ATTACH ':memory:' AS other", "PRAGMA journal_mode=WAL"):
                with self.subTest(query=query), runtime.connection(self.path) as connection:
                    with self.assertRaises(apsw.AuthError):
                        connection.execute(query).fetchall()
            with runtime.connection(self.path) as connection:
                with self.assertRaises(apsw.SQLError):
                    connection.execute("SELECT absent_column FROM values_table").fetchall()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_invalid_runtime_is_rejected_instead_of_silently_changing_grades(self):
        for replacement in (dict(sqlite_version="0.0.0"), dict(evaluation_time_utc="2026-10-02"),
                            dict(vm_steps_per_query=999), dict(vm_steps_per_query=0)):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                FrozenSQLite(**(self.settings | replacement))


if __name__ == "__main__":
    unittest.main()
