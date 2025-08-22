-- Sample users
INSERT OR IGNORE INTO users (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com');

-- Sample places
INSERT INTO places (name, category, district) VALUES
('Mihrimah Sultan Mosque', 'Mosque', 'Kadikoy'),
('Maidens Tower', 'Historical Place', 'Uskudar'),
('Archeology Museum', 'Museum', 'Fatih'),
('Nuruosmaniye Mosque', 'Mosque', 'Fatih'),
('Grand Bazaar', 'Historical Place', 'Fatih'),
('Basilica Cistern', 'Historical Place', 'Fatih'),
('Hagia Sophia Grand Mosque', 'Mosque', 'Fatih'),
('Turkish & Islamic Arts Museum', 'Museum', 'Fatih'),
('The Blue Mosque', 'Mosque', 'Fatih'),
('Sultanahmet Square', 'Historical Place', 'Fatih'),
('Topkapi Palace Museum', 'Museum', 'Fatih'),
('Istanbul Museum of Modern Art', 'Museum', 'Beyoglu'),
('Pera Museum', 'Museum', 'Beyoglu'),
('Church of Saint Anthony of Padua', 'Church', 'Beyoglu'),
('Çiçek Pasajı', 'Historical Place', 'Beyoglu'),
('Hagia Triada Greek Orthodox Church', 'Church', 'Beyoglu'),
('Galata Tower', 'Historical Place', 'Beyoglu'),
('Taksim Square', 'District', 'Beyoglu'),
('Istiklal Avenue', 'District', 'Beyoglu'),
('Cagaloglu Hammam', 'Historical Place', 'Fatih'),
('Egyptian Bazaar', 'Historical Place', 'Fatih'),
('Suleymaniye Mosque', 'Mosque', 'Fatih'),
('Museum Of Illusions Istanbul', 'Museum', 'Beyoglu'),
('Venerable Patriarchal Church of Saint George', 'Church', 'Fatih'),
('Dolmabahçe Palace', 'Museum', 'Besiktas'),
('National Painting Museum', 'Museum', 'Besiktas'),
('Naval Museum', 'Museum', 'Besiktas'),
('National Palaces Depot', 'Museum', 'Besiktas'),
('Büyük Mecidiye Mosque', 'Mosque', 'Besiktas'),
('Yıldız Park', 'Park', 'Besiktas'),
('Yıldız Hamidiye Mosque', 'Mosque', 'Besiktas'),
('Nisantasi', 'District', 'Sisli'),
('Harbiye Military Museum', 'Museum', 'Sisli'),
('Yapı Kredi Kültür Sanat Museum', 'Museum', 'Beyoglu'),
('Roman Catholic Church of Santa Maria Draperis', 'Church', 'Beyoglu'),
('Taksim Mosque', 'Mosque', 'Beyoglu'),
('Rahmi M. Koç Museum', 'Museum', 'Beyoglu'),
('İBB Maçka Democracy Park', 'Park', 'Besiktas');

-- Sample events
INSERT INTO events (name, venue, date) VALUES
('Istanbul Jazz Festival', 'Harbiye', '2025-09-15 19:00'),
('Art Exhibition', 'Istanbul Modern', '2025-08-30 10:00');
