#!/bin/bash
set -e  # Остановиться при ошибке

# install Postgres
apt-get update
apt-get install -y postgresql postgresql-contrib

# Prepare data directory
mkdir -p /var/lib/postgresql/14/main
rm -rf /var/lib/postgresql/14/main/*
chown -R postgres:postgres /var/lib/postgresql/14/main

# Init DB
su - postgres -c "/usr/lib/postgresql/14/bin/initdb -D /var/lib/postgresql/14/main"

# Start server (wait until ready)
su - postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l logfile start"
sleep 3  # небольшая пауза, чтобы сервер успел подняться

# Configuration
DB_NAME="oneactor"
DB_USER="bot_user"
DB_PASSWORD="123456"

# Create DB and user
su - postgres -c "psql -v ON_ERROR_STOP=1 --username=postgres <<EOF
CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF"
