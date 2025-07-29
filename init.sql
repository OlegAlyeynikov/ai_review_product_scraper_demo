-- PostgreSQL initialization script
-- This file will be executed when the database container starts

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

SELECT 'PostgreSQL initialization completed successfully' AS status;
