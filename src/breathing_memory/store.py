from __future__ import annotations

import datetime as dt
from pathlib import Path
import shutil
import sqlite3
from typing import Optional

from .models import Anchor, Fragment, FragmentFeedback, FragmentReference, SequenceMetric


LEGACY_TABLES = {
    "memory_fragments",
    "reference_logs",
    "feedback_logs",
    "runtime_state",
}

REQUIRED_TABLES = {
    "anchors",
    "fragments",
    "fragment_references",
    "fragment_feedback",
    "sequence_metrics",
}


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_database_file()
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def close(self) -> None:
        self.connection.close()

    def _prepare_database_file(self) -> None:
        if not self.db_path.exists():
            return
        connection = sqlite3.connect(str(self.db_path))
        try:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        finally:
            connection.close()
        table_names = {row[0] for row in rows}
        if not table_names:
            return
        is_legacy = bool(table_names & LEGACY_TABLES)
        missing_required = not REQUIRED_TABLES.issubset(table_names)
        if not is_legacy and not missing_required:
            return
        backup = self._legacy_backup_path()
        shutil.move(str(self.db_path), str(backup))

    def _legacy_backup_path(self) -> Path:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        suffix = self.db_path.suffix or ".sqlite3"
        return self.db_path.with_name(f"{self.db_path.stem}.legacy-{timestamp}{suffix}")

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS anchors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                replies_to_anchor_id INTEGER NULL REFERENCES anchors(id) ON DELETE SET NULL,
                is_root INTEGER NOT NULL CHECK (is_root IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
                parent_id INTEGER NULL REFERENCES fragments(id) ON DELETE SET NULL,
                actor TEXT NOT NULL CHECK (actor IN ('user', 'agent')),
                content TEXT NOT NULL,
                content_length INTEGER NOT NULL,
                embedding_vector BLOB NULL,
                layer TEXT NOT NULL CHECK (layer IN ('working', 'holding')),
                compression_fail_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS fragment_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
                fragment_id INTEGER NOT NULL REFERENCES fragments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS fragment_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
                fragment_id INTEGER NOT NULL REFERENCES fragments(id) ON DELETE CASCADE,
                verdict TEXT NOT NULL CHECK (verdict IN ('positive', 'neutral', 'negative'))
            );

            CREATE TABLE IF NOT EXISTS sequence_metrics (
                anchor_id INTEGER PRIMARY KEY REFERENCES anchors(id) ON DELETE CASCADE,
                working_usage_bytes INTEGER NOT NULL,
                holding_usage_bytes INTEGER NOT NULL,
                compress_count INTEGER NOT NULL,
                delete_count INTEGER NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS purge_orphan_anchor_after_fragment_delete
            AFTER DELETE ON fragments
            FOR EACH ROW
            BEGIN
                DELETE FROM anchors
                WHERE id = OLD.anchor_id
                  AND NOT EXISTS (
                    SELECT 1 FROM fragments WHERE anchor_id = OLD.anchor_id
                  );
            END;
            """
        )
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_anchors_replies_to_anchor_id
            ON anchors(replies_to_anchor_id, id);

            CREATE INDEX IF NOT EXISTS idx_fragments_anchor_id
            ON fragments(anchor_id, id);

            CREATE INDEX IF NOT EXISTS idx_fragments_parent_id
            ON fragments(parent_id, id);

            CREATE INDEX IF NOT EXISTS idx_fragments_layer
            ON fragments(layer, id);

            CREATE INDEX IF NOT EXISTS idx_fragment_references_from_anchor_id
            ON fragment_references(from_anchor_id, id);

            CREATE INDEX IF NOT EXISTS idx_fragment_references_fragment_id
            ON fragment_references(fragment_id, id);

            CREATE INDEX IF NOT EXISTS idx_fragment_feedback_from_anchor_id
            ON fragment_feedback(from_anchor_id, id);

            CREATE INDEX IF NOT EXISTS idx_fragment_feedback_fragment_id
            ON fragment_feedback(fragment_id, id);
            """
        )
        self.connection.commit()

    def create_anchor(self, replies_to_anchor_id: Optional[int], is_root: bool) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO anchors (replies_to_anchor_id, is_root)
            VALUES (?, ?)
            """,
            (replies_to_anchor_id, 1 if is_root else 0),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def create_fragment(
        self,
        anchor_id: int,
        actor: str,
        content: str,
        layer: str,
        parent_id: Optional[int] = None,
        embedding_vector: Optional[bytes] = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO fragments (
                anchor_id,
                parent_id,
                actor,
                content,
                content_length,
                embedding_vector,
                layer,
                compression_fail_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                anchor_id,
                parent_id,
                actor,
                content,
                self.content_size(content),
                embedding_vector,
                layer,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def update_fragment_embedding(self, fragment_id: int, embedding_vector: Optional[bytes]) -> None:
        self.connection.execute(
            """
            UPDATE fragments
            SET embedding_vector = ?
            WHERE id = ?
            """,
            (embedding_vector, fragment_id),
        )
        self.connection.commit()

    def create_reference(self, from_anchor_id: int, fragment_id: int) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO fragment_references (from_anchor_id, fragment_id)
            VALUES (?, ?)
            """,
            (from_anchor_id, fragment_id),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def create_feedback(self, from_anchor_id: int, fragment_id: int, verdict: str) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO fragment_feedback (from_anchor_id, fragment_id, verdict)
            VALUES (?, ?, ?)
            """,
            (from_anchor_id, fragment_id, verdict),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def copy_references(self, source_fragment_id: int, target_fragment_id: int) -> None:
        rows = self.connection.execute(
            """
            SELECT from_anchor_id
            FROM fragment_references
            WHERE fragment_id = ?
            ORDER BY id ASC
            """,
            (source_fragment_id,),
        ).fetchall()
        for row in rows:
            self.connection.execute(
                """
                INSERT INTO fragment_references (from_anchor_id, fragment_id)
                VALUES (?, ?)
                """,
                (row["from_anchor_id"], target_fragment_id),
            )
        self.connection.commit()

    def copy_feedback(self, source_fragment_id: int, target_fragment_id: int) -> None:
        rows = self.connection.execute(
            """
            SELECT from_anchor_id, verdict
            FROM fragment_feedback
            WHERE fragment_id = ?
            ORDER BY id ASC
            """,
            (source_fragment_id,),
        ).fetchall()
        for row in rows:
            self.connection.execute(
                """
                INSERT INTO fragment_feedback (from_anchor_id, fragment_id, verdict)
                VALUES (?, ?, ?)
                """,
                (row["from_anchor_id"], target_fragment_id, row["verdict"]),
            )
        self.connection.commit()

    def record_sequence_metrics(
        self,
        anchor_id: int,
        working_usage_bytes: int,
        holding_usage_bytes: int,
        compress_count: int,
        delete_count: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO sequence_metrics (
                anchor_id,
                working_usage_bytes,
                holding_usage_bytes,
                compress_count,
                delete_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                anchor_id,
                working_usage_bytes,
                holding_usage_bytes,
                compress_count,
                delete_count,
            ),
        )
        self.connection.commit()

    def list_anchors(self) -> list[Anchor]:
        rows = self.connection.execute(
            "SELECT * FROM anchors ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_anchor(row) for row in rows]

    def list_fragments(self) -> list[Fragment]:
        rows = self.connection.execute(
            "SELECT * FROM fragments ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def list_fragments_by_anchor(self, anchor_id: int) -> list[Fragment]:
        rows = self.connection.execute(
            """
            SELECT * FROM fragments
            WHERE anchor_id = ?
            ORDER BY id ASC
            """,
            (anchor_id,),
        ).fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def list_root_fragments_replying_to_anchor(self, reply_to_anchor_id: int, actor: str) -> list[Fragment]:
        rows = self.connection.execute(
            """
            SELECT fragments.*
            FROM fragments
            INNER JOIN anchors ON anchors.id = fragments.anchor_id
            WHERE anchors.replies_to_anchor_id = ?
              AND fragments.actor = ?
              AND fragments.parent_id IS NULL
            ORDER BY fragments.id DESC
            """,
            (reply_to_anchor_id, actor),
        ).fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def list_references(self) -> list[FragmentReference]:
        rows = self.connection.execute(
            "SELECT * FROM fragment_references ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_reference(row) for row in rows]

    def list_references_from_anchor(self, from_anchor_id: int) -> list[FragmentReference]:
        rows = self.connection.execute(
            """
            SELECT * FROM fragment_references
            WHERE from_anchor_id = ?
            ORDER BY id ASC
            """,
            (from_anchor_id,),
        ).fetchall()
        return [self._row_to_reference(row) for row in rows]

    def list_references_for_fragment(self, fragment_id: int) -> list[FragmentReference]:
        rows = self.connection.execute(
            """
            SELECT * FROM fragment_references
            WHERE fragment_id = ?
            ORDER BY id ASC
            """,
            (fragment_id,),
        ).fetchall()
        return [self._row_to_reference(row) for row in rows]

    def list_feedback(self) -> list[FragmentFeedback]:
        rows = self.connection.execute(
            "SELECT * FROM fragment_feedback ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_feedback(row) for row in rows]

    def list_feedback_for_fragment(self, fragment_id: int) -> list[FragmentFeedback]:
        rows = self.connection.execute(
            """
            SELECT * FROM fragment_feedback
            WHERE fragment_id = ?
            ORDER BY id ASC
            """,
            (fragment_id,),
        ).fetchall()
        return [self._row_to_feedback(row) for row in rows]

    def list_sequence_metrics(self) -> list[SequenceMetric]:
        rows = self.connection.execute(
            "SELECT * FROM sequence_metrics ORDER BY anchor_id ASC"
        ).fetchall()
        return [self._row_to_sequence_metric(row) for row in rows]

    def get_anchor(self, anchor_id: int) -> Optional[Anchor]:
        row = self.connection.execute(
            "SELECT * FROM anchors WHERE id = ?",
            (anchor_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_anchor(row)

    def get_fragment(self, fragment_id: int) -> Optional[Fragment]:
        row = self.connection.execute(
            "SELECT * FROM fragments WHERE id = ?",
            (fragment_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_fragment(row)

    def update_fragment_layer(self, fragment_id: int, layer: str) -> None:
        self.connection.execute(
            """
            UPDATE fragments
            SET layer = ?
            WHERE id = ?
            """,
            (layer, fragment_id),
        )
        self.connection.commit()

    def increment_compression_fail_count(self, fragment_id: int, amount: int = 1) -> None:
        self.connection.execute(
            """
            UPDATE fragments
            SET compression_fail_count = compression_fail_count + ?
            WHERE id = ?
            """,
            (amount, fragment_id),
        )
        self.connection.commit()

    def delete_fragment(self, fragment_id: int) -> None:
        row = self.connection.execute(
            """
            SELECT parent_id
            FROM fragments
            WHERE id = ?
            """,
            (fragment_id,),
        ).fetchone()
        if row is None:
            return
        parent_id = row["parent_id"]
        if parent_id is not None and self.fragment_exists(parent_id):
            self.increment_compression_fail_count(parent_id, amount=1)
        self.connection.execute(
            "DELETE FROM fragments WHERE id = ?",
            (fragment_id,),
        )
        self.connection.commit()

    def anchor_exists(self, anchor_id: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM anchors WHERE id = ? LIMIT 1",
            (anchor_id,),
        ).fetchone()
        return row is not None

    def fragment_exists(self, fragment_id: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM fragments WHERE id = ? LIMIT 1",
            (fragment_id,),
        ).fetchone()
        return row is not None

    def max_anchor_sequence(self) -> int:
        row = self.connection.execute(
            """
            SELECT seq
            FROM sqlite_sequence
            WHERE name = 'anchors'
            """
        ).fetchone()
        if row is None or row["seq"] is None:
            return 0
        return int(row["seq"])

    def current_anchor_id(self) -> int:
        row = self.connection.execute(
            "SELECT MAX(id) AS value FROM anchors"
        ).fetchone()
        if row is None or row["value"] is None:
            return 0
        return int(row["value"])

    def _row_to_anchor(self, row: sqlite3.Row) -> Anchor:
        return Anchor(
            id=int(row["id"]),
            replies_to_anchor_id=row["replies_to_anchor_id"],
            is_root=bool(row["is_root"]),
        )

    def _row_to_fragment(self, row: sqlite3.Row) -> Fragment:
        return Fragment(
            id=int(row["id"]),
            anchor_id=int(row["anchor_id"]),
            parent_id=row["parent_id"],
            actor=row["actor"],
            content=row["content"],
            content_length=int(row["content_length"]),
            embedding_vector=row["embedding_vector"],
            layer=row["layer"],
            compression_fail_count=int(row["compression_fail_count"]),
        )

    def _row_to_reference(self, row: sqlite3.Row) -> FragmentReference:
        return FragmentReference(
            id=int(row["id"]),
            from_anchor_id=int(row["from_anchor_id"]),
            fragment_id=int(row["fragment_id"]),
        )

    def _row_to_feedback(self, row: sqlite3.Row) -> FragmentFeedback:
        return FragmentFeedback(
            id=int(row["id"]),
            from_anchor_id=int(row["from_anchor_id"]),
            fragment_id=int(row["fragment_id"]),
            verdict=row["verdict"],
        )

    def _row_to_sequence_metric(self, row: sqlite3.Row) -> SequenceMetric:
        return SequenceMetric(
            anchor_id=int(row["anchor_id"]),
            working_usage_bytes=int(row["working_usage_bytes"]),
            holding_usage_bytes=int(row["holding_usage_bytes"]),
            compress_count=int(row["compress_count"]),
            delete_count=int(row["delete_count"]),
        )

    @staticmethod
    def content_size(content: str) -> int:
        return len(content.encode("utf-8"))
