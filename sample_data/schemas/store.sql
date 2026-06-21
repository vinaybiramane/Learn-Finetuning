CREATE TABLE customers (
    id   INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount      REAL,
    status      TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
