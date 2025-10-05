"""
Query Clustering and Template Generation System for AI Istanbul
Automatically clusters user queries and generates template responses
Targets 95% coverage of FAQ-like queries without GPT dependency
"""

import json
import pickle
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import re
import hashlib

# ML libraries
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Advanced libraries (optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class QueryCluster:
    """Represents a cluster of similar queries"""
    cluster_id: str
    name: str
    description: str
    intent_type: str
    
    # Query patterns
    query_patterns: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)
    
    # Template information
    template_id: str = ""
    template_variants: List[str] = field(default_factory=list)
    
    # Statistics
    query_count: int = 0
    avg_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Context requirements
    required_context: List[str] = field(default_factory=list)
    optional_context: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'cluster_id': self.cluster_id,
            'name': self.name,
            'description': self.description,
            'intent_type': self.intent_type,
            'query_patterns': self.query_patterns,
            'example_queries': self.example_queries,
            'template_id': self.template_id,
            'template_variants': self.template_variants,
            'query_count': self.query_count,
            'avg_confidence': self.avg_confidence,
            'last_updated': self.last_updated.isoformat(),
            'required_context': self.required_context,
            'optional_context': self.optional_context
        }

@dataclass
class ResponseTemplate:
    """Template for generating responses"""
    template_id: str
    cluster_id: str
    template_text: str
    
    # Variables and placeholders
    variables: List[str] = field(default_factory=list)
    context_dependencies: List[str] = field(default_factory=list)
    
    # Variations for different contexts
    variants: Dict[str, str] = field(default_factory=dict)
    
    # Performance metrics
    usage_count: int = 0
    user_satisfaction: float = 0.8
    last_used: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'template_id': self.template_id,
            'cluster_id': self.cluster_id,
            'template_text': self.template_text,
            'variables': self.variables,
            'context_dependencies': self.context_dependencies,
            'variants': self.variants,
            'usage_count': self.usage_count,
            'user_satisfaction': self.user_satisfaction,
            'last_used': self.last_used.isoformat()
        }

class QueryClusteringSystem:
    """
    Advanced query clustering and template generation system
    Uses ML to identify query patterns and auto-generate responses
    """
    
    def __init__(self, data_dir: str = "clustering_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Clustering components
        self.clusters: Dict[str, QueryCluster] = {}
        self.templates: Dict[str, ResponseTemplate] = {}
        self.query_history: List[Dict] = []
        
        # ML components
        self.embedding_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True,
            min_df=2
        )
        
        # Clustering parameters
        self.min_queries_per_cluster = 5
        self.similarity_threshold = 0.7
        self.max_clusters = 50
        
        # Statistics
        self.stats = {
            'total_queries_processed': 0,
            'clusters_generated': 0,
            'templates_created': 0,
            'successful_matches': 0,
            'last_clustering': None
        }
        
        # Initialize components
        self._initialize_ml_components()
        self._load_existing_data()
        self._initialize_predefined_clusters()
    
    def _initialize_ml_components(self):
        """Initialize ML models"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded SentenceTransformer model")
            else:
                logger.warning("⚠️ Using TF-IDF for query embeddings")
        except Exception as e:
            logger.error(f"❌ Error initializing ML components: {e}")
    
    def _initialize_predefined_clusters(self):
        """Initialize common Istanbul query clusters"""
        predefined_clusters = [
            {
                'name': 'Transportation Directions',
                'intent_type': 'transportation',
                'patterns': [
                    'how to get to {location}',
                    'best way to {location}',
                    'metro to {location}',
                    'bus to {location}',
                    'transport to {location}'
                ],
                'template': """🚇 **Getting to {location}:**

**Metro Route:**
{metro_route}

**Bus Options:**
{bus_options}

**Walking Time:** {walking_time}
**Estimated Cost:** {cost}

💡 *Pro tip: Use an Istanbul Card for discounted public transport!*"""
            },
            {
                'name': 'Opening Hours and Tickets',
                'intent_type': 'practical_info',
                'patterns': [
                    '{attraction} opening hours',
                    '{attraction} tickets',
                    '{attraction} price',
                    'when does {attraction} open',
                    'how much is {attraction}'
                ],
                'template': """🎫 **{attraction} Information:**

**Opening Hours:**
{opening_hours}

**Ticket Prices:**
{ticket_prices}

**Best Time to Visit:** {best_time}
**Duration:** {visit_duration}

📱 *Book online to skip the queues!*"""
            },
            {
                'name': 'Restaurant Recommendations',
                'intent_type': 'food',
                'patterns': [
                    'best restaurants in {area}',
                    'where to eat in {area}',
                    'good food near {location}',
                    '{cuisine} restaurants',
                    'traditional turkish food'
                ],
                'template': """🍽️ **Restaurant Recommendations for {area}:**

**Top Picks:**
{restaurant_list}

**Local Specialties:**
{local_dishes}

**Price Range:** {price_range}
**Reservations:** {reservation_info}

🌟 *Don't miss: {must_try_dish}*"""
            },
            {
                'name': 'Area Overview and Highlights',
                'intent_type': 'exploration',
                'patterns': [
                    'what to see in {area}',
                    'things to do in {area}',
                    '{area} attractions',
                    'visit {area}',
                    'explore {area}'
                ],
                'template': """🏛️ **Exploring {area}:**

**Must-See Attractions:**
{main_attractions}

**Hidden Gems:**
{hidden_gems}

**Getting Around:**
{transport_info}

**Perfect for:** {visitor_type}
**Time Needed:** {time_required}

🎯 *Insider tip: {local_tip}*"""
            },
            {
                'name': 'Shopping and Markets',
                'intent_type': 'shopping',
                'patterns': [
                    'shopping in {area}',
                    'best markets',
                    'where to buy souvenirs',
                    'grand bazaar',
                    'local shops'
                ],
                'template': """🛍️ **Shopping in {area}:**

**Best Markets:**
{markets}

**What to Buy:**
{products}

**Bargaining Tips:**
{bargaining_tips}

**Opening Hours:** {hours}

💰 *Budget: {budget_guide}*"""
            }
        ]
        
        for cluster_def in predefined_clusters:
            cluster_id = self._generate_cluster_id(cluster_def['name'])
            
            cluster = QueryCluster(
                cluster_id=cluster_id,
                name=cluster_def['name'],
                description=f"Queries about {cluster_def['name'].lower()}",
                intent_type=cluster_def['intent_type'],
                query_patterns=cluster_def['patterns']
            )
            
            # Create template
            template_id = f"template_{cluster_id}"
            template = ResponseTemplate(
                template_id=template_id,
                cluster_id=cluster_id,
                template_text=cluster_def['template']
            )
            
            cluster.template_id = template_id
            self.clusters[cluster_id] = cluster
            self.templates[template_id] = template
        
        logger.info(f"✅ Initialized {len(predefined_clusters)} predefined clusters")
    
    def add_query_sample(self, query: str, response: str, intent_type: str, 
                        context: Dict = None, user_satisfaction: float = 0.8):
        """Add a query sample for clustering analysis"""
        query_sample = {
            'query': query.lower().strip(),
            'response': response,
            'intent_type': intent_type,
            'context': context or {},
            'user_satisfaction': user_satisfaction,
            'timestamp': datetime.now().isoformat()
        }
        
        self.query_history.append(query_sample)
        self.stats['total_queries_processed'] += 1
        
        # Trigger clustering if we have enough samples
        if len(self.query_history) % 100 == 0:
            self._perform_clustering()
    
    def add_query(self, query: str, response: str = "", intent_type: str = "general", 
                  context: Dict = None, user_satisfaction: float = 0.8):
        """Alias for add_query_sample for backward compatibility"""
        return self.add_query_sample(query, response, intent_type, context, user_satisfaction)
    
    def _perform_clustering(self):
        """Perform query clustering on collected samples"""
        try:
            if len(self.query_history) < self.min_queries_per_cluster * 2:
                return
            
            logger.info(f"🔄 Starting clustering on {len(self.query_history)} queries...")
            
            # Extract queries
            queries = [sample['query'] for sample in self.query_history[-1000:]]  # Use recent queries
            
            # Generate embeddings
            if self.embedding_model:
                embeddings = self.embedding_model.encode(queries)
            else:
                # Fallback to TF-IDF
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(queries)
                embeddings = tfidf_matrix.toarray()
            
            # Determine optimal number of clusters
            optimal_clusters = self._find_optimal_clusters(embeddings, max_k=min(20, len(queries)//5))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Analyze clusters
            self._analyze_clusters(queries, cluster_labels, embeddings)
            
            self.stats['last_clustering'] = datetime.now().isoformat()
            logger.info(f"✅ Clustering completed: {optimal_clusters} clusters generated")
            
        except Exception as e:
            logger.error(f"❌ Error during clustering: {e}")
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """Find optimal number of clusters using silhouette analysis"""
        try:
            if len(embeddings) < 4:
                return 2
            
            silhouette_scores = []
            k_range = range(2, min(max_k, len(embeddings)) + 1)
            
            for k in k_range:
                if k >= len(embeddings):
                    break
                    
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Skip if all points in one cluster
                if len(set(cluster_labels)) < 2:
                    continue
                    
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append((k, silhouette_avg))
            
            if not silhouette_scores:
                return 3
            
            # Find k with highest silhouette score
            optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
            return optimal_k
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {e}")
            return 5  # Default fallback
    
    def _analyze_clusters(self, queries: List[str], labels: np.ndarray, embeddings: np.ndarray):
        """Analyze generated clusters and create templates"""
        try:
            cluster_groups = defaultdict(list)
            for query, label in zip(queries, labels):
                cluster_groups[label].append(query)
            
            for cluster_id, cluster_queries in cluster_groups.items():
                if len(cluster_queries) < self.min_queries_per_cluster:
                    continue
                
                # Analyze cluster patterns
                cluster_info = self._analyze_cluster_patterns(cluster_queries)
                
                # Generate cluster object
                new_cluster = QueryCluster(
                    cluster_id=f"auto_{cluster_id}_{int(datetime.now().timestamp())}",
                    name=cluster_info['name'],
                    description=cluster_info['description'],
                    intent_type=cluster_info['intent_type'],
                    query_patterns=cluster_info['patterns'],
                    example_queries=cluster_queries[:5],
                    query_count=len(cluster_queries)
                )
                
                # Generate template
                template = self._generate_template_for_cluster(new_cluster, cluster_queries)
                if template:
                    new_cluster.template_id = template.template_id
                    self.templates[template.template_id] = template
                
                self.clusters[new_cluster.cluster_id] = new_cluster
                self.stats['clusters_generated'] += 1
                
                logger.info(f"📊 Generated cluster: {new_cluster.name} ({len(cluster_queries)} queries)")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing clusters: {e}")
    
    def _analyze_cluster_patterns(self, queries: List[str]) -> Dict:
        """Analyze patterns in a cluster of queries"""
        # Extract common words and patterns
        all_words = []
        for query in queries:
            words = re.findall(r'\b\w+\b', query.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        common_words = [word for word, freq in word_freq.most_common(10) if freq > 1]
        
        # Determine intent type based on keywords
        intent_keywords = {
            'transportation': ['get', 'go', 'metro', 'bus', 'transport', 'way', 'reach'],
            'food': ['restaurant', 'eat', 'food', 'dinner', 'lunch', 'breakfast', 'cuisine'],
            'practical_info': ['hours', 'open', 'close', 'price', 'ticket', 'cost', 'when'],
            'exploration': ['see', 'visit', 'do', 'attractions', 'places', 'explore'],
            'shopping': ['buy', 'shop', 'market', 'bazaar', 'souvenir', 'store']
        }
        
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for word in common_words if word in keywords)
            intent_scores[intent] = score
        
        best_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'general'
        
        # Generate name and description
        key_terms = common_words[:3]
        name = f"Queries about {' and '.join(key_terms)}"
        description = f"Cluster containing queries related to {', '.join(key_terms)}"
        
        # Extract patterns
        patterns = self._extract_query_patterns(queries)
        
        return {
            'name': name,
            'description': description,
            'intent_type': best_intent,
            'patterns': patterns,
            'common_words': common_words
        }
    
    def _extract_query_patterns(self, queries: List[str]) -> List[str]:
        """Extract common patterns from queries"""
        patterns = []
        
        # Find common phrase structures
        for query in queries[:5]:  # Analyze first 5 queries
            # Replace specific locations/names with placeholders
            pattern = query
            
            # Common Istanbul locations
            location_replacements = {
                'hagia sophia': '{attraction}',
                'blue mosque': '{attraction}',
                'galata tower': '{attraction}',
                'grand bazaar': '{attraction}',
                'sultanahmet': '{area}',
                'beyoglu': '{area}',
                'taksim': '{area}'
            }
            
            for location, placeholder in location_replacements.items():
                if location in pattern:
                    pattern = pattern.replace(location, placeholder)
            
            patterns.append(pattern)
        
        return list(set(patterns))  # Remove duplicates
    
    def _generate_template_for_cluster(self, cluster: QueryCluster, 
                                     sample_queries: List[str]) -> Optional[ResponseTemplate]:
        """Generate a response template for a cluster"""
        try:
            template_id = f"template_{cluster.cluster_id}"
            
            # Generate template based on intent type
            template_generators = {
                'transportation': self._generate_transport_template,
                'food': self._generate_food_template,
                'practical_info': self._generate_info_template,
                'exploration': self._generate_exploration_template,
                'shopping': self._generate_shopping_template
            }
            
            generator = template_generators.get(cluster.intent_type, self._generate_generic_template)
            template_text = generator(cluster, sample_queries)
            
            if template_text:
                template = ResponseTemplate(
                    template_id=template_id,
                    cluster_id=cluster.cluster_id,
                    template_text=template_text,
                    variables=self._extract_template_variables(template_text)
                )
                
                self.stats['templates_created'] += 1
                return template
            
        except Exception as e:
            logger.error(f"❌ Error generating template: {e}")
        
        return None
    
    def _generate_transport_template(self, cluster: QueryCluster, queries: List[str]) -> str:
        """Generate template for transportation queries"""
        return """🚇 **Getting to {destination}:**

**Metro Route:**
{metro_route}

**Bus Options:**
{bus_options} 

**Walking Distance:** {walking_time}
**Total Cost:** {estimated_cost}

💡 *Pro tip: {transport_tip}*

📱 Download Citymapper or Moovit for real-time transport updates!"""

    def _generate_food_template(self, cluster: QueryCluster, queries: List[str]) -> str:
        """Generate template for food-related queries"""
        return """🍽️ **Food Recommendations:**

**Top Restaurants:**
{restaurant_recommendations}

**Local Specialties:**
{local_dishes}

**Price Range:** {price_range}
**Best Time:** {best_dining_time}

🌟 *Must try: {signature_dish}*

📍 *Location tips: {location_advice}*"""

    def _generate_info_template(self, cluster: QueryCluster, queries: List[str]) -> str:
        """Generate template for practical information queries"""
        return """ℹ️ **Practical Information:**

**Hours:** {opening_hours}
**Prices:** {ticket_prices}
**Duration:** {visit_duration}

**Tips:**
{practical_tips}

📱 *Book ahead: {booking_info}*"""

    def _generate_exploration_template(self, cluster: QueryCluster, queries: List[str]) -> str:
        """Generate template for exploration queries"""
        return """🗺️ **Explore {area}:**

**Must-See:**
{main_attractions}

**Hidden Gems:**
{hidden_spots}

**Perfect for:** {visitor_type}
**Time needed:** {duration}

🎯 *Local tip: {insider_advice}*"""

    def _generate_shopping_template(self, cluster: QueryCluster, queries: List[str]) -> str:
        """Generate template for shopping queries"""
        return """🛍️ **Shopping Guide:**

**Best Places:**
{shopping_locations}

**What to Buy:**
{products}

**Bargaining:** {bargaining_tips}
**Budget:** {price_guide}

💰 *Pro tip: {shopping_advice}*"""

    def _generate_generic_template(self, cluster: QueryCluster, queries: List[str]) -> str:
        """Generate generic template for unclassified queries"""
        return """💡 **About {topic}:**

{information}

**Helpful Details:**
{details}

**Recommendations:**
{suggestions}

ℹ️ *Additional info: {extra_info}*"""

    def _extract_template_variables(self, template_text: str) -> List[str]:
        """Extract variables from template text"""
        import re
        variables = re.findall(r'\{(\w+)\}', template_text)
        return list(set(variables))
    
    def _generate_cluster_id(self, name: str) -> str:
        """Generate unique cluster ID"""
        return hashlib.md5(name.encode()).hexdigest()[:12]
    
    def match_query_to_cluster(self, query: str, context: Dict = None) -> Optional[Tuple[QueryCluster, float]]:
        """Match a query to the best cluster"""
        try:
            query_normalized = query.lower().strip()
            best_match = None
            best_score = 0.0
            
            for cluster in self.clusters.values():
                # Calculate similarity with patterns
                pattern_scores = []
                for pattern in cluster.query_patterns:
                    pattern_normalized = pattern.lower()
                    score = self._calculate_pattern_similarity(query_normalized, pattern_normalized)
                    pattern_scores.append(score)
                
                # Calculate similarity with example queries
                example_scores = []
                for example in cluster.example_queries:
                    example_normalized = example.lower()
                    score = self._calculate_text_similarity(query_normalized, example_normalized)
                    example_scores.append(score)
                
                # Combine scores
                max_pattern_score = max(pattern_scores) if pattern_scores else 0.0
                max_example_score = max(example_scores) if example_scores else 0.0
                combined_score = max(max_pattern_score, max_example_score)
                
                # Context bonus
                if context and cluster.intent_type in context.get('preferred_types', []):
                    combined_score += 0.1
                
                if combined_score > best_score and combined_score >= self.similarity_threshold:
                    best_score = combined_score
                    best_match = cluster
            
            if best_match:
                self.stats['successful_matches'] += 1
                return best_match, best_score
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error matching query to cluster: {e}")
            return None
    
    def _calculate_pattern_similarity(self, query: str, pattern: str) -> float:
        """Calculate similarity between query and pattern"""
        # Handle placeholder patterns
        if '{' in pattern and '}' in pattern:
            # Convert pattern to regex
            regex_pattern = re.escape(pattern)
            regex_pattern = regex_pattern.replace(r'\{[^}]+\}', r'.*')
            
            if re.match(regex_pattern, query):
                return 0.9  # High score for pattern match
        
        # Fallback to word overlap
        query_words = set(query.split())
        pattern_words = set(pattern.replace('{', '').replace('}', '').split())
        
        if not pattern_words:
            return 0.0
        
        overlap = len(query_words.intersection(pattern_words))
        return overlap / len(pattern_words)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_template_response(self, cluster: QueryCluster, query: str, 
                            context: Dict = None) -> Optional[str]:
        """Generate response using cluster template"""
        try:
            if not cluster.template_id or cluster.template_id not in self.templates:
                return None
            
            template = self.templates[cluster.template_id]
            template_text = template.template_text
            
            # Fill in template variables with context or defaults
            variables = template.variables
            filled_template = template_text
            
            for var in variables:
                value = self._get_variable_value(var, query, context, cluster)
                filled_template = filled_template.replace(f'{{{var}}}', value)
            
            # Update usage statistics
            template.usage_count += 1
            template.last_used = datetime.now()
            
            return filled_template
            
        except Exception as e:
            logger.error(f"❌ Error generating template response: {e}")
            return None
    
    def get_template_for_query(self, query: str, context: Dict = None) -> Optional[str]:
        """Get template response for a query by finding best cluster match"""
        try:
            # Find best matching cluster
            cluster_id, confidence = self.find_best_cluster(query, context)
            
            if cluster_id and cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                return self.get_template_response(cluster, query, context)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting template for query: {e}")
            return None
    
    def _get_variable_value(self, variable: str, query: str, context: Dict, 
                          cluster: QueryCluster) -> str:
        """Get value for template variable"""
        # Extract from context first
        if context and variable in context:
            return str(context[variable])
        
        # Try to extract from query
        extracted_value = self._extract_from_query(variable, query)
        if extracted_value:
            return extracted_value
        
        # Default values based on variable type
        defaults = {
            'destination': 'your destination',
            'location': 'the location',
            'area': 'this area',
            'attraction': 'this attraction',
            'metro_route': 'Take the M1 or M2 metro line',
            'bus_options': 'Multiple bus routes available',
            'walking_time': '10-15 minutes',
            'estimated_cost': '5-10 TL',
            'transport_tip': 'Use Istanbul Card for best rates',
            'restaurant_recommendations': 'Several excellent options nearby',
            'local_dishes': 'Turkish breakfast, kebabs, and baklava',
            'price_range': 'Budget to mid-range options',
            'opening_hours': '9:00 AM - 6:00 PM (typical)',
            'ticket_prices': 'Check official website for current prices',
            'practical_tips': 'Visit early to avoid crowds'
        }
        
        return defaults.get(variable, f'[{variable}]')
    
    def _extract_from_query(self, variable: str, query: str) -> Optional[str]:
        """Try to extract variable value from query text"""
        query_lower = query.lower()
        
        # Location extraction patterns
        if variable in ['destination', 'location', 'area', 'attraction']:
            istanbul_locations = {
                'hagia sophia': 'Hagia Sophia',
                'blue mosque': 'Blue Mosque',
                'galata tower': 'Galata Tower',
                'grand bazaar': 'Grand Bazaar',
                'sultanahmet': 'Sultanahmet',
                'beyoglu': 'Beyoğlu',
                'taksim': 'Taksim Square',
                'bosphorus': 'Bosphorus',
                'topkapi': 'Topkapi Palace'
            }
            
            for key, value in istanbul_locations.items():
                if key in query_lower:
                    return value
        
        return None
    
    def _load_existing_data(self):
        """Load existing clusters and templates from disk"""
        try:
            clusters_file = self.data_dir / "clusters.json"
            templates_file = self.data_dir / "templates.json"
            
            if clusters_file.exists():
                with open(clusters_file, 'r') as f:
                    clusters_data = json.load(f)
                    for cluster_id, cluster_dict in clusters_data.items():
                        cluster_dict['last_updated'] = datetime.fromisoformat(cluster_dict['last_updated'])
                        self.clusters[cluster_id] = QueryCluster(**cluster_dict)
            
            if templates_file.exists():
                with open(templates_file, 'r') as f:
                    templates_data = json.load(f)
                    for template_id, template_dict in templates_data.items():
                        template_dict['last_used'] = datetime.fromisoformat(template_dict['last_used'])
                        self.templates[template_id] = ResponseTemplate(**template_dict)
            
            logger.info(f"✅ Loaded {len(self.clusters)} clusters and {len(self.templates)} templates")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing data: {e}")
    
    def save_data(self):
        """Save clusters and templates to disk"""
        try:
            clusters_file = self.data_dir / "clusters.json"
            templates_file = self.data_dir / "templates.json"
            
            # Save clusters
            clusters_data = {cid: cluster.to_dict() for cid, cluster in self.clusters.items()}
            with open(clusters_file, 'w') as f:
                json.dump(clusters_data, f, indent=2)
            
            # Save templates
            templates_data = {tid: template.to_dict() for tid, template in self.templates.items()}
            with open(templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            logger.info(f"💾 Saved {len(self.clusters)} clusters and {len(self.templates)} templates")
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
    
    def get_cluster_statistics(self) -> Dict:
        """Get clustering system statistics"""
        return {
            'total_clusters': len(self.clusters),
            'total_templates': len(self.templates),
            'clustering_stats': self.stats,
            'cluster_types': {
                intent: len([c for c in self.clusters.values() if c.intent_type == intent])
                for intent in ['transportation', 'food', 'practical_info', 'exploration', 'shopping', 'general']
            },
            'template_usage': {
                tid: template.usage_count 
                for tid, template in self.templates.items()
            }
        }

# Integration class for seamless usage
class GPTFreeQueryProcessor:
    """
    Main processor that uses clustering system to handle queries without GPT
    Integrates with existing AI Istanbul components
    """
    
    def __init__(self, clustering_system: QueryClusteringSystem, 
                 semantic_cache=None, query_router=None):
        self.clustering_system = clustering_system
        self.semantic_cache = semantic_cache
        self.query_router = query_router
        
        # Fallback handling
        self.fallback_responses = {
            'transportation': "I can help you get around Istanbul! Please specify your destination and I'll provide transport options.",
            'food': "Istanbul has amazing food! Let me know what area you're interested in and I'll recommend great restaurants.",
            'practical_info': "I can provide practical information about Istanbul attractions. What would you like to know?",
            'exploration': "Istanbul has so much to explore! Tell me about your interests and I'll suggest places to visit.",
            'shopping': "Istanbul is a shopper's paradise! Let me know what you're looking for and I'll guide you to the best places."
        }
    
    def process_query(self, query: str, context: Dict = None, user_id: str = None) -> Dict:
        """Process query using clustering system first, then fallbacks"""
        try:
            # Step 1: Try semantic cache if available
            if self.semantic_cache:
                cached_result = self.semantic_cache.get_cached_response(query, context, user_id)
                if cached_result:
                    response, confidence, metadata = cached_result
                    return {
                        'response': response,
                        'source': 'semantic_cache',
                        'confidence': confidence,
                        'metadata': metadata
                    }
            
            # Step 2: Try query clustering
            cluster_match = self.clustering_system.match_query_to_cluster(query, context)
            if cluster_match:
                cluster, confidence = cluster_match
                template_response = self.clustering_system.get_template_response(cluster, query, context)
                
                if template_response:
                    # Add to cache for future use
                    if self.semantic_cache:
                        self.semantic_cache.add_to_cache(
                            query, template_response, cluster.intent_type, 
                            'en', context, cluster.template_id
                        )
                    
                    return {
                        'response': template_response,
                        'source': 'query_clustering',
                        'confidence': confidence,
                        'metadata': {
                            'cluster_id': cluster.cluster_id,
                            'cluster_name': cluster.name,
                            'template_id': cluster.template_id,
                            'intent_type': cluster.intent_type
                        }
                    }
            
            # Step 3: Try existing query router if available
            if self.query_router:
                try:
                    router_result = self.query_router.route_query(query, context or {})
                    if router_result and router_result.get('response'):
                        return {
                            'response': router_result['response'],
                            'source': 'query_router',
                            'confidence': router_result.get('confidence', 0.7),
                            'metadata': router_result.get('metadata', {})
                        }
                except Exception as e:
                    logger.warning(f"Query router failed: {e}")
            
            # Step 4: Fallback response
            # Try to determine intent for appropriate fallback
            query_lower = query.lower()
            intent_keywords = {
                'transportation': ['get', 'go', 'metro', 'bus', 'transport', 'way'],
                'food': ['restaurant', 'eat', 'food', 'dinner', 'lunch'],
                'practical_info': ['hours', 'open', 'price', 'ticket', 'cost'],
                'exploration': ['see', 'visit', 'do', 'attractions', 'places'],
                'shopping': ['buy', 'shop', 'market', 'bazaar', 'souvenir']
            }
            
            detected_intent = 'general'
            max_matches = 0
            
            for intent, keywords in intent_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in query_lower)
                if matches > max_matches:
                    max_matches = matches
                    detected_intent = intent
            
            fallback_response = self.fallback_responses.get(
                detected_intent, 
                "I'd be happy to help you explore Istanbul! Could you please be more specific about what you're looking for?"
            )
            
            return {
                'response': fallback_response,
                'source': 'fallback',
                'confidence': 0.3,
                'metadata': {
                    'detected_intent': detected_intent,
                    'reason': 'no_matching_cluster'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                'response': "I apologize, but I'm having trouble processing your request right now. Please try again or contact support.",
                'source': 'error',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
