def create_tables(connection) -> None:
    if getattr(connection, "dialect", "sqlite") == "mysql":
        statements = (
            """CREATE TABLE IF NOT EXISTS datasets (
                dataset_id VARCHAR(32) PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_path TEXT NOT NULL,
                processed_path TEXT NOT NULL,
                `rows` BIGINT NOT NULL,
                `columns` INT NOT NULL,
                uploaded_at VARCHAR(40) NOT NULL,
                updated_at VARCHAR(40) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE TABLE IF NOT EXISTS history (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                dataset_id VARCHAR(32) NOT NULL,
                operation VARCHAR(255) NOT NULL,
                column_name TEXT,
                rows_before BIGINT NOT NULL,
                rows_after BIGINT NOT NULL,
                details TEXT NOT NULL,
                timestamp VARCHAR(40) NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            """CREATE INDEX idx_history_dataset_time
                ON history(dataset_id, timestamp DESC)""",
        )
        for statement in statements:
            try:
                connection.execute(statement)
            except Exception as error:
                if "Duplicate key name" not in str(error):
                    raise
        return

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            processed_path TEXT NOT NULL,
            `rows` INTEGER NOT NULL,
            `columns` INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            operation TEXT NOT NULL,
            column_name TEXT,
            rows_before INTEGER NOT NULL,
            rows_after INTEGER NOT NULL,
            details TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_history_dataset_time
            ON history(dataset_id, timestamp DESC);
        """
    )
