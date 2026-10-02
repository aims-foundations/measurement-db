"""Read-only SQL execution with a recorded clock, engine, and work allowance."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from uuid import uuid4

import apsw


class FrozenSQLite(apsw.VFS):
    """Override SQLite's clock without rewriting SQL or changing database bytes.

    Each connection is for one query. Its progress handler interrupts after a
    fixed number of VM instructions; machine speed cannot change that allowance.
    The context sets the process time zone to UTC for SQL's 'localtime' modifier
    and restores it on exit. Use one runtime per curation process. The wrapper
    leaves execution and result comparison to callers.
    """

    def __init__(self, *, sqlite_version, evaluation_time_utc, vm_steps_per_query, localtime_timezone="UTC"):
        if apsw.sqlitelibversion() != sqlite_version:
            raise ValueError(f"SQLite {sqlite_version} is required; found {apsw.sqlitelibversion()}")
        if localtime_timezone != "UTC":
            raise ValueError("The reproducible SQL runtime requires localtime_timezone: UTC")
        instant = datetime.fromisoformat(evaluation_time_utc.replace("Z", "+00:00"))
        if instant.utcoffset() is None or instant.utcoffset().total_seconds() != 0:
            raise ValueError("The SQL evaluation clock must have an explicit UTC offset")
        if isinstance(vm_steps_per_query, bool) or not isinstance(vm_steps_per_query, int) or (
                vm_steps_per_query <= 0 or vm_steps_per_query % 1000):
            raise ValueError("The SQL VM allowance must be a positive multiple of 1,000")
        # SQLite's VFS uses milliseconds since the Julian epoch.
        self.clock_ms = round(instant.astimezone(timezone.utc).timestamp() * 1000) + 210866760000000
        self.max_callbacks = vm_steps_per_query // 1000
        self.name = "measurement-db-" + uuid4().hex
        super().__init__(self.name, "")

    def xCurrentTime(self):
        return self.clock_ms / 86400000

    def xCurrentTimeInt64(self):
        return self.clock_ms

    def __enter__(self):
        self.previous_timezone = os.environ.get("TZ")
        os.environ["TZ"] = "UTC"
        time.tzset()
        return self

    def __exit__(self, *_):
        self.unregister()
        if self.previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = self.previous_timezone
        time.tzset()

    @contextmanager
    def connection(self, database):
        # These captured databases cannot change during evaluation. Immutable
        # access also avoids creating WAL shared-memory files beside the inputs.
        connection = apsw.Connection(Path(database).resolve().as_uri() + "?immutable=1",
                                     flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI, vfs=self.name)
        allowed = {apsw.SQLITE_SELECT, apsw.SQLITE_READ, apsw.SQLITE_FUNCTION, apsw.SQLITE_RECURSIVE}
        connection.set_authorizer(lambda action, *_: apsw.SQLITE_OK if action in allowed else apsw.SQLITE_DENY)
        callbacks = 0

        def exhausted():
            nonlocal callbacks
            callbacks += 1
            return callbacks >= self.max_callbacks

        connection.set_progress_handler(exhausted, 1000)
        try:
            yield connection
        finally:
            connection.close()
