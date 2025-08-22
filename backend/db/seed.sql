-- Sample users
INSERT OR IGNORE INTO users (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com');

-- Sample places
INSERT INTO places (name, category, lat, lng) VALUES
('Galata Tower', 'Historic', 41.0256, 28.9744),
('Istiklal Street', 'Shopping', 41.0359, 28.9850),
('Sunny Fenerbahce Park', 'Kadikoy', NULL, NULL),
('Sunny Caddebostan seaside', 'Kadikoy', NULL, NULL),
('Bagdat Avenue', 'Kadikoy', NULL, NULL),
('Moda', 'Kadikoy', NULL, NULL),
('Kadikoy District', 'Kadikoy', NULL, NULL),
('Mihrimah Sultan Mosque', 'Uskudar', NULL, NULL),
('Maiden''s Tower', 'Uskudar', NULL, NULL),
('Archeology Museum', 'Fatih', NULL, NULL),
('Nuruosmaniye Mosque', 'Fatih', NULL, NULL),
('Cagaloglu Hammam', 'Fatih', NULL, NULL),
('Egyptian Bazaar', 'Fatih', NULL, NULL),
('Suleymaniye Mosque', 'Fatih', NULL, NULL),
('Grand Bazaar', 'Fatih', NULL, NULL),
('Basilica Cistern', 'Fatih', NULL, NULL),
('Hagia Sophia Grand Mosque', 'Fatih', NULL, NULL),
('Turkish & Islamic Arts Museum', 'Fatih', NULL, NULL),
('The Blue Mosque', 'Fatih', NULL, NULL),
('Sultanahmet Square', 'Fatih', NULL, NULL),
('Topkapi Palace Museum', 'Fatih', NULL, NULL),
('Istanbul Museum of Modern Art', 'Beyoglu', NULL, NULL),
('Pera Museum', 'Beyoglu', NULL, NULL),
('Church of Saint Anthony of Padua', 'Beyoglu', NULL, NULL),
('Çiçek Pasajı', 'Beyoglu', NULL, NULL),
('Hagia Triada Greek Orthodox Church', 'Beyoglu', NULL, NULL),
('Galata Tower', 'Beyoglu', NULL, NULL),
('Taksim Square', 'Beyoglu', NULL, NULL),
('Istiklal Avenue', 'Beyoglu', NULL, NULL),
('Museum Of Illusions Istanbul', 'Beyoglu', NULL, NULL),
('Venerable Patriarchal Church of Saint George', 'Fatih', NULL, NULL),
('Dolmabahçe Palace Museum', 'Besiktas', NULL, NULL),
('National Painting Museum', 'Besiktas', NULL, NULL),
('Naval Museum', 'Besiktas', NULL, NULL),
('National Palaces Depot Museum', 'Besiktas', NULL, NULL),
('Büyük Mecidiye Mosque', 'Besiktas', NULL, NULL),
('Yıldız Park', 'Besiktas', NULL, NULL),
('Yıldız Hamidiye Mosque', 'Besiktas', NULL, NULL),
('Nisantasi', 'Sisli', NULL, NULL),
('Harbiye Military Museum', 'Sisli', NULL, NULL),
('Yapı Kredi Kültür Sanat Museum', 'Beyoglu', NULL, NULL),
('Roman Catholic Church of Santa Maria Draperis', 'Beyoglu', NULL, NULL),
('Taksim Mosque', 'Beyoglu', NULL, NULL),
('Rahmi M. Koç Museum', 'Beyoglu', NULL, NULL),
('İBB Maçka Democracy Park', 'Sisli', NULL, NULL);

-- Sample events
INSERT INTO events (title, venue, date) VALUES
('Istanbul Jazz Festival', 'Harbiye', '2025-09-15 19:00'),
('Art Exhibition', 'Istanbul Modern', '2025-08-30 10:00');
