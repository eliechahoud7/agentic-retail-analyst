-- Read-only inspection of retail_intelligence. Run with default_transaction_read_only=on.
-- Each query is separated by a line containing only "-- @@" so a runner can label results.

-- @@ session
SELECT current_user, session_user, current_database(),
       current_setting('default_transaction_read_only') AS read_only,
       current_setting('server_version') AS server_version;

-- @@ role_attributes
SELECT rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin,
       rolreplication, rolbypassrls, rolconnlimit, rolvaliduntil
FROM pg_roles WHERE rolname = current_user;

-- @@ role_memberships (direct and inherited)
WITH RECURSIVE m AS (
  SELECT oid, rolname FROM pg_roles WHERE rolname = current_user
  UNION
  SELECT r.oid, r.rolname FROM pg_auth_members am
  JOIN m ON am.member = m.oid JOIN pg_roles r ON r.oid = am.roleid
)
SELECT rolname FROM m WHERE rolname <> current_user;

-- @@ database_privileges
SELECT has_database_privilege(current_database(), 'CONNECT') AS can_connect,
       has_database_privilege(current_database(), 'CREATE')  AS can_create_schema,
       has_database_privilege(current_database(), 'TEMP')    AS can_create_temp;

-- @@ schemas
SELECT n.nspname AS schema, pg_get_userbyid(n.nspowner) AS owner,
       has_schema_privilege(n.nspname, 'USAGE')  AS usage,
       has_schema_privilege(n.nspname, 'CREATE') AS can_create
FROM pg_namespace n
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
  AND n.nspname NOT LIKE 'pg_temp_%' AND n.nspname NOT LIKE 'pg_toast_temp_%'
ORDER BY 1;

-- @@ relations_and_privileges
SELECT n.nspname AS schema, c.relname AS name,
       CASE c.relkind WHEN 'r' THEN 'table' WHEN 'v' THEN 'view' WHEN 'm' THEN 'matview'
                      WHEN 'p' THEN 'partitioned' WHEN 'f' THEN 'foreign' END AS kind,
       c.reltuples::bigint AS est_rows,
       has_table_privilege(c.oid, 'SELECT') AS sel,
       has_table_privilege(c.oid, 'INSERT') AS ins,
       has_table_privilege(c.oid, 'UPDATE') AS upd,
       has_table_privilege(c.oid, 'DELETE') AS del,
       has_table_privilege(c.oid, 'TRUNCATE') AS trunc,
       has_table_privilege(c.oid, 'REFERENCES') AS refs,
       has_table_privilege(c.oid, 'TRIGGER') AS trig
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'v', 'm', 'p', 'f')
  AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg_toast%'
ORDER BY 1, 2;

-- @@ columns
SELECT c.table_schema AS schema, c.table_name AS tbl, c.ordinal_position AS pos,
       c.column_name AS col, c.data_type, c.is_nullable AS nullable
FROM information_schema.columns c
WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1, 2, 3;

-- @@ primary_and_unique_keys
SELECT n.nspname AS schema, t.relname AS tbl, con.conname, con.contype,
       pg_get_constraintdef(con.oid) AS definition
FROM pg_constraint con
JOIN pg_class t ON t.oid = con.conrelid JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE con.contype IN ('p', 'u') AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1, 2;

-- @@ foreign_keys
SELECT n.nspname AS schema, t.relname AS tbl, con.conname,
       pg_get_constraintdef(con.oid) AS definition
FROM pg_constraint con
JOIN pg_class t ON t.oid = con.conrelid JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE con.contype = 'f' AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1, 2;

-- @@ view_definitions
SELECT schemaname AS schema, viewname AS name, definition
FROM pg_views WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
UNION ALL
SELECT schemaname, matviewname, definition
FROM pg_matviews WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1, 2;
