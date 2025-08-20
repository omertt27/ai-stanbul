-- Sample users
INSERT INTO users (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com');

-- Sample places
INSERT INTO places (name, category, lat, lng) VALUES
('Galata Tower', 'Historic', 41.0256, 28.9744),
('Istiklal Street', 'Shopping', 41.0359, 28.9850);

-- Sample events
INSERT INTO events (title, venue, date) VALUES
('Istanbul Jazz Festival', 'Harbiye', '2025-09-15 19:00'),
('Art Exhibition', 'Istanbul Modern', '2025-08-30 10:00');