--
-- PostgreSQL database dump
--

\restrict hv3IH62ThRFDsBaEgxogkwBPAxTcLYKVNvBtz4kPhCqx0uxoQUBo0uWAyy40DAq

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.7 (Homebrew)

-- Started on 2025-12-11 15:24:12 +03

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 223 (class 1259 OID 16445)
-- Name: blog_posts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.blog_posts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 236 (class 1259 OID 16458)
-- Name: blog_posts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.blog_posts (
    id integer DEFAULT nextval('public.blog_posts_id_seq'::regclass) NOT NULL,
    title character varying(200) NOT NULL,
    content text NOT NULL,
    author character varying(100),
    district character varying(100),
    created_at timestamp without time zone,
    likes_count integer
);


--
-- TOC entry 222 (class 1259 OID 16444)
-- Name: chat_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 237 (class 1259 OID 16481)
-- Name: chat_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_history (
    id integer DEFAULT nextval('public.chat_history_id_seq'::regclass) NOT NULL,
    session_id character varying(100) NOT NULL,
    user_message text NOT NULL,
    ai_response text NOT NULL,
    "timestamp" timestamp without time zone
);


--
-- TOC entry 233 (class 1259 OID 16455)
-- Name: chat_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_sessions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 238 (class 1259 OID 16488)
-- Name: chat_sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_sessions (
    id integer DEFAULT nextval('public.chat_sessions_id_seq'::regclass) NOT NULL,
    session_id character varying(100) NOT NULL,
    user_id character varying(100),
    started_at timestamp without time zone,
    last_activity timestamp without time zone,
    messages_count integer,
    active_navigation_session character varying(100),
    has_navigation boolean,
    context json,
    is_active boolean,
    ended_at timestamp without time zone
);


--
-- TOC entry 234 (class 1259 OID 16456)
-- Name: conversation_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.conversation_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 239 (class 1259 OID 16497)
-- Name: conversation_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.conversation_history (
    id integer DEFAULT nextval('public.conversation_history_id_seq'::regclass) NOT NULL,
    session_id character varying(100) NOT NULL,
    user_id character varying(100),
    user_message text NOT NULL,
    ai_response text NOT NULL,
    route_data json,
    location_data json,
    navigation_active boolean,
    "timestamp" timestamp without time zone,
    intent character varying(100),
    entities_extracted json
);


--
-- TOC entry 221 (class 1259 OID 16443)
-- Name: events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 240 (class 1259 OID 16506)
-- Name: events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.events (
    id integer DEFAULT nextval('public.events_id_seq'::regclass) NOT NULL,
    name character varying,
    venue character varying,
    date timestamp without time zone,
    genre character varying,
    biletix_id character varying
);


--
-- TOC entry 224 (class 1259 OID 16446)
-- Name: feedback_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.feedback_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 241 (class 1259 OID 16515)
-- Name: feedback_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.feedback_events (
    id integer DEFAULT nextval('public.feedback_events_id_seq'::regclass) NOT NULL,
    user_id character varying(100) NOT NULL,
    session_id character varying(100),
    event_type character varying(50) NOT NULL,
    item_id character varying(100) NOT NULL,
    item_type character varying(50) NOT NULL,
    metadata json,
    "timestamp" timestamp without time zone,
    processed boolean
);


--
-- TOC entry 228 (class 1259 OID 16450)
-- Name: intent_feedback_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.intent_feedback_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 242 (class 1259 OID 16527)
-- Name: intent_feedback; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.intent_feedback (
    id integer DEFAULT nextval('public.intent_feedback_id_seq'::regclass) NOT NULL,
    session_id character varying(100) NOT NULL,
    user_id character varying(100),
    original_query text NOT NULL,
    language character varying(10),
    predicted_intent character varying(50) NOT NULL,
    predicted_confidence double precision NOT NULL,
    classification_method character varying(20),
    latency_ms double precision,
    is_correct boolean,
    actual_intent character varying(50),
    feedback_type character varying(20),
    "timestamp" timestamp without time zone NOT NULL,
    user_action character varying(100),
    used_for_training boolean,
    review_status character varying(20),
    context_data text
);


--
-- TOC entry 226 (class 1259 OID 16448)
-- Name: item_feature_vectors_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.item_feature_vectors_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 243 (class 1259 OID 16546)
-- Name: item_feature_vectors; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.item_feature_vectors (
    id integer DEFAULT nextval('public.item_feature_vectors_id_seq'::regclass) NOT NULL,
    item_id character varying(100) NOT NULL,
    item_type character varying(50) NOT NULL,
    embedding_vector json,
    embedding_version character varying(50),
    total_views integer,
    total_clicks integer,
    total_saves integer,
    avg_rating double precision,
    conversion_rate double precision,
    quality_score double precision,
    updated_at timestamp without time zone
);


--
-- TOC entry 229 (class 1259 OID 16451)
-- Name: location_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.location_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 244 (class 1259 OID 16554)
-- Name: location_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.location_history (
    id integer DEFAULT nextval('public.location_history_id_seq'::regclass) NOT NULL,
    user_id character varying(100) NOT NULL,
    session_id character varying(100),
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    accuracy double precision,
    altitude double precision,
    speed double precision,
    heading double precision,
    "timestamp" timestamp without time zone,
    activity_type character varying(50),
    is_navigation_active boolean
);


--
-- TOC entry 219 (class 1259 OID 16441)
-- Name: museums_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.museums_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 245 (class 1259 OID 16562)
-- Name: museums; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.museums (
    id integer DEFAULT nextval('public.museums_id_seq'::regclass) NOT NULL,
    name character varying,
    location character varying,
    hours character varying,
    ticket_price double precision,
    highlights character varying
);


--
-- TOC entry 231 (class 1259 OID 16453)
-- Name: navigation_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.navigation_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 246 (class 1259 OID 16570)
-- Name: navigation_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.navigation_events (
    id integer DEFAULT nextval('public.navigation_events_id_seq'::regclass) NOT NULL,
    session_id character varying(100) NOT NULL,
    user_id character varying(100) NOT NULL,
    event_type character varying(50) NOT NULL,
    event_data json,
    latitude double precision,
    longitude double precision,
    current_step integer,
    step_instruction text,
    distance_to_next_step double precision,
    "timestamp" timestamp without time zone
);


--
-- TOC entry 230 (class 1259 OID 16452)
-- Name: navigation_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.navigation_sessions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 247 (class 1259 OID 16580)
-- Name: navigation_sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.navigation_sessions (
    id integer DEFAULT nextval('public.navigation_sessions_id_seq'::regclass) NOT NULL,
    session_id character varying(100) NOT NULL,
    user_id character varying(100) NOT NULL,
    chat_session_id character varying(100),
    origin_lat double precision NOT NULL,
    origin_lon double precision NOT NULL,
    origin_name character varying(255),
    destination_lat double precision NOT NULL,
    destination_lon double precision NOT NULL,
    destination_name character varying(255),
    waypoints json,
    total_distance double precision,
    total_duration double precision,
    transport_mode character varying(50),
    current_step_index integer,
    steps_completed integer,
    distance_remaining double precision,
    time_remaining double precision,
    status character varying(50),
    is_active boolean,
    route_geometry json,
    route_steps json,
    started_at timestamp without time zone,
    completed_at timestamp without time zone,
    last_update timestamp without time zone,
    actual_duration double precision,
    deviations_count integer,
    reroutes_count integer
);


--
-- TOC entry 227 (class 1259 OID 16449)
-- Name: online_learning_models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.online_learning_models_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 248 (class 1259 OID 16590)
-- Name: online_learning_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.online_learning_models (
    id integer DEFAULT nextval('public.online_learning_models_id_seq'::regclass) NOT NULL,
    model_name character varying(100) NOT NULL,
    model_version character varying(50) NOT NULL,
    model_type character varying(50) NOT NULL,
    parameters json,
    hyperparameters json,
    metrics json,
    is_active boolean,
    is_deployed boolean,
    deployment_percentage double precision,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


--
-- TOC entry 218 (class 1259 OID 16440)
-- Name: places_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.places_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 249 (class 1259 OID 16597)
-- Name: places; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.places (
    id integer DEFAULT nextval('public.places_id_seq'::regclass) NOT NULL,
    name character varying(255) NOT NULL,
    category character varying(50),
    district character varying(50)
);


--
-- TOC entry 220 (class 1259 OID 16442)
-- Name: restaurants_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.restaurants_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 250 (class 1259 OID 16601)
-- Name: restaurants; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.restaurants (
    id integer DEFAULT nextval('public.restaurants_id_seq'::regclass) NOT NULL,
    name character varying,
    cuisine character varying,
    location character varying,
    rating double precision,
    source character varying,
    description character varying,
    place_id character varying,
    phone character varying,
    website character varying,
    price_level integer,
    photo_url text,
    photo_reference text
);


--
-- TOC entry 235 (class 1259 OID 16457)
-- Name: route_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.route_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 251 (class 1259 OID 16610)
-- Name: route_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.route_history (
    id integer DEFAULT nextval('public.route_history_id_seq'::regclass) NOT NULL,
    user_id character varying(100) NOT NULL,
    navigation_session_id character varying(100),
    origin character varying(255),
    destination character varying(255),
    waypoints json,
    distance double precision,
    duration double precision,
    transport_mode character varying(50),
    route_geometry json,
    steps json,
    user_rating integer,
    user_feedback text,
    completed_at timestamp without time zone
);


--
-- TOC entry 225 (class 1259 OID 16447)
-- Name: user_interaction_aggregates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.user_interaction_aggregates_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 252 (class 1259 OID 16618)
-- Name: user_interaction_aggregates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_interaction_aggregates (
    id integer DEFAULT nextval('public.user_interaction_aggregates_id_seq'::regclass) NOT NULL,
    user_id character varying(100) NOT NULL,
    item_type character varying(50) NOT NULL,
    view_count integer,
    click_count integer,
    save_count integer,
    rating_count integer,
    conversion_count integer,
    avg_rating double precision,
    avg_dwell_time double precision,
    click_through_rate double precision,
    conversion_rate double precision,
    last_interaction timestamp without time zone,
    recency_score double precision,
    category_preferences json,
    updated_at timestamp without time zone
);


--
-- TOC entry 232 (class 1259 OID 16454)
-- Name: user_preferences_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.user_preferences_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 253 (class 1259 OID 16626)
-- Name: user_preferences; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_preferences (
    id integer DEFAULT nextval('public.user_preferences_id_seq'::regclass) NOT NULL,
    user_id character varying(100) NOT NULL,
    preferred_transport character varying(50),
    avoid_highways boolean,
    avoid_tolls boolean,
    avoid_ferries boolean,
    wheelchair_accessible boolean,
    requires_elevator boolean,
    preferred_language character varying(10),
    distance_units character varying(10),
    voice_guidance boolean,
    notification_sound boolean,
    vibration boolean,
    save_location_history boolean,
    share_location boolean,
    interests json,
    dietary_restrictions json,
    budget_level character varying(20),
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


--
-- TOC entry 217 (class 1259 OID 16439)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.users_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 254 (class 1259 OID 16633)
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id integer DEFAULT nextval('public.users_id_seq'::regclass) NOT NULL,
    name character varying(100) NOT NULL,
    email character varying(100) NOT NULL
);


--
-- TOC entry 4474 (class 0 OID 16458)
-- Dependencies: 236
-- Data for Name: blog_posts; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.blog_posts (id, title, content, author, district, created_at, likes_count) FROM stdin;
1	Hidden Gems of Istanbul: Secret Rooftop Gardens and Terraces	# Discover Istanbul's Secret Sky Gardens\n\nWhile tourists flock to famous landmarks, Istanbul's most magical experiences often happen above the crowds. The city's hidden rooftop gardens and terraces offer breathtaking views, peaceful moments, and unique perspectives on this ancient metropolis.\n\n## � Secret Garden Terraces\n\n### Galata Mevlevi Lodge Garden\nA hidden oasis in the heart of Galata:\n- **Location**: Behind the Whirling Dervish Museum\n- **Features**: 500-year-old trees, meditation corners\n- **Views**: Golden Horn and Old City panorama\n- **Best Time**: Early morning or sunset\n- **Access**: Small entrance fee, often overlooked by tourists\n\n### Yıldız Park Secret Terraces\nOttoman imperial gardens with hidden viewing spots:\n- **History**: Former palace grounds of Sultan Abdulhamid II\n- **Hidden Spots**: Malta Köşkü upper terrace, Çadır Köşkü gardens\n- **Features**: Century-old magnolias, secret pathways\n- **Views**: Bosphorus through ancient trees\n- **Pro Tip**: Visit during weekdays for solitude\n\n## 🏛️ Historic Building Rooftops\n\n### Süleymaniye Mosque Courtyard Terraces\nElevated spaces around Sinan's masterpiece:\n- **Architecture**: 16th-century Ottoman design\n- **Views**: Golden Horn, Galata Tower, modern Istanbul\n- **Atmosphere**: Spiritual tranquility above city chaos\n- **Features**: Original marble terraces, historic fountains\n- **Cultural Note**: Respectful behavior required\n\n### Pierre Loti Hill Tea Gardens\nMultiple levels of terraced gardens:\n- **Transport**: Historic cable car or steep walk\n- **Levels**: Three different elevation terraces\n- **Views**: 180-degree Golden Horn panorama\n- **Traditional**: Ottoman-style tea service\n- **Sunset**: One of city's most romantic spots\n\n## 🍃 Modern Rooftop Sanctuaries\n\n### Karaköy Rooftop Gardens\nIndustrial district's green transformation:\n- **Character**: Converted warehouse rooftops\n- **Features**: Urban farming, artist studios\n- **Views**: Bosphorus Bridge, modern city skyline\n- **Culture**: Local art installations, pop-up events\n- **Access**: Some require local knowledge to find\n\n### Kadıköy Hidden Terraces\nAsian side's secret elevated spaces:\n- **Moda Terraces**: Converted apartment building rooftops\n- **Features**: Community gardens, reading nooks\n- **Views**: Prince Islands, European side skyline\n- **Atmosphere**: Local, authentic, unhurried\n- **Best**: Weekend morning coffee culture\n\n## 🌅 Best Times and Seasons\n\n### Golden Hour Magic\n- **Sunrise**: 6:00-7:30 AM (April-September)\n- **Sunset**: 6:00-8:00 PM (varies by season)\n- **Light**: Soft, warm illumination of city\n- **Photography**: Perfect conditions for memorable shots\n\n### Seasonal Considerations\n- **Spring**: Blooming trees, mild temperatures\n- **Summer**: Long days, warm evenings\n- **Autumn**: Clear air, stunning colors\n- **Winter**: Dramatic skies, fewer crowds\n\n## 📍 How to Find Hidden Spots\n\n### Local Intelligence\n- **Ask locals**: Neighborhood tea house owners\n- **Social media**: Instagram location tags\n- **Walking**: Explore residential areas\n- **Observation**: Look up while walking\n\n### Access Tips\n- **Respect**: Many are private or semi-private\n- **Timing**: Avoid prayer times near mosques\n- **Dress**: Modest clothing for religious sites\n- **Behavior**: Quiet, respectful presence\n\n## 🎯 Photography and Etiquette\n\n### Best Shots\n- **Wide angles**: Capture full city panoramas\n- **Details**: Ancient architectural elements\n- **Layers**: Foreground gardens, background cityscape\n- **People**: Ask permission before photographing locals\n\n### Respectful Behavior\n- **Noise**: Keep voices low\n- **Litter**: Leave no trace\n- **Privacy**: Respect residents and worshippers\n- **Cultural sensitivity**: Follow local customs\n\nThese hidden elevated spaces offer a different perspective on Istanbul—literally and figuratively. Take time to discover these peaceful sanctuaries above the bustling streets, and you'll find some of the city's most memorable moments.\n\n*Have you discovered any secret rooftop spots in Istanbul? Share your hidden gems in the comments below!*	Urban Explorer	\N	2025-09-23 22:10:00	2
2	Istanbul's Coffee Culture Revolution: From Traditional to Third Wave	# The Evolution of Istanbul's Coffee Scene\n\nIstanbul's relationship with coffee spans centuries, from the Ottoman Empire's first coffee houses to today's artisanal third-wave movement. This journey through the city's coffee culture reveals how tradition and innovation create something uniquely Turkish.\n\n## ☕ Historical Foundation\n\n### Ottoman Coffee Heritage\nCoffee arrived in Istanbul in the 16th century:\n- **Origin**: Brought from Yemen by Ottoman traders\n- **Innovation**: World's first coffee houses opened here\n- **Culture**: Social centers for politics, poetry, chess\n- **UNESCO**: Turkish coffee inscribed as cultural heritage\n- **Preparation**: Sand-roasted, served with Turkish delight\n\n### Traditional Preparation Method\n**Turkish Coffee (Türk Kahvesi)**:\n- **Grind**: Powder-fine, finer than espresso\n- **Pot**: Special brass or copper cezve/ibrik\n- **Heat**: Slow cooking over sand or low flame\n- **Foam**: Crucial creamy foam on top\n- **Serving**: Small cups with traditional sweets\n\n## �️ Historic Coffee Houses\n\n### Fazıl Bey Turkish Coffee\nPreserving 19th-century traditions:\n- **Location**: Kadıköy, Asian side authenticity\n- **Founded**: 1923, four generations of coffee masters\n- **Specialty**: 40+ varieties of Turkish coffee\n- **Atmosphere**: Original Ottoman interior, brass equipment\n- **Experience**: Coffee fortune telling (tasseography)\n\n### Şark Kahvesi\nGrand Bazaar's historic gem:\n- **History**: Operating since 1944\n- **Location**: Inside the covered Grand Bazaar\n- **Character**: Unchanged traditional atmosphere\n- **Clientele**: Local merchants, savvy tourists\n- **Specialty**: Traditional Turkish coffee, backgammon\n\n### Corlulu Ali Paşa Medresesi\nCoffee in a 16th-century courtyard:\n- **Setting**: Ottoman theological school courtyard\n- **Architecture**: Classical Ottoman stone architecture\n- **Atmosphere**: Nargile (hookah) and traditional coffee\n- **Crowd**: University students, elderly locals\n- **Unique**: Coffee served in historic religious setting\n\n## ☕ Modern Third Wave Movement\n\n### Artisanal Coffee Pioneers\n**Coffee Sapiens** (Karaköy):\n- **Philosophy**: Science meets artistry\n- **Equipment**: Professional espresso machines, pour-over stations\n- **Beans**: Single-origin, locally roasted\n- **Education**: Barista workshops, cupping sessions\n- **Innovation**: Turkish coffee with modern techniques\n\n**Petra Roasting Co.** (Galata):\n- **Focus**: Small-batch roasting, traceability\n- **Location**: Historic Galata district\n- **Offerings**: Pour-over, cold brew, espresso\n- **Community**: Regular coffee education events\n- **Quality**: Competition-grade coffee preparation\n\n### Fusion Concepts\n**Walter's Coffee Roastery**:\n- **Concept**: Turkish tradition meets global methods\n- **Locations**: Multiple districts across Istanbul\n- **Menu**: Traditional Turkish coffee alongside specialty drinks\n- **Innovation**: Turkish coffee ice cream, modern interpretations\n- **Design**: Contemporary spaces honoring coffee heritage\n\n## 🌍 Neighborhood Coffee Cultures\n\n### Beyoğlu/Galata Scene\nBohemian coffee culture:\n- **Character**: Artistic, international, experimental\n- **Venues**: Independent roasters, specialty cafes\n- **Crowd**: Creative professionals, expats, students\n- **Innovation**: Fusion drinks, alternative brewing methods\n- **Atmosphere**: Relaxed, work-friendly spaces\n\n### Kadıköy Alternative Culture\nAsian side authenticity:\n- **Vibe**: Local, unpretentious, community-focused\n- **Quality**: High standards without pretension\n- **Price**: More affordable than European side\n- **Innovation**: Experimental brewing, local roasters\n- **Social**: Strong regular customer communities\n\n### Nişantaşı Luxury Scene\nUpscale coffee experiences:\n- **Style**: Elegant, refined, premium\n- **Venues**: Designer coffee shops, luxury hotels\n- **Price**: Premium pricing for premium experience\n- **Service**: White-glove coffee service\n- **Clientele**: Business professionals, luxury shoppers\n\n## 🔬 Coffee Preparation Techniques\n\n### Traditional Methods Still Popular\n**Turkish Coffee**:\n- **Time**: 3-5 minutes careful preparation\n- **Skill**: Years to master perfect foam\n- **Ritual**: Ceremonial aspect still important\n- **Social**: Shared experience, conversation catalyst\n\n### Modern Brewing Methods\n**Pour-Over Revolution**:\n- **Equipment**: V60, Chemex, Aeropress\n- **Focus**: Bean origin, extraction precision\n- **Time**: 3-4 minutes, careful technique\n- **Result**: Clean, bright flavors highlighting bean character\n\n**Espresso Culture**:\n- **Quality**: Italian-level equipment and training\n- **Standards**: Precise timing, temperature, pressure\n- **Innovation**: Turkish-inspired espresso drinks\n- **Accessibility**: Quick service for busy city life\n\n## 🏆 Best Coffee Experiences\n\n### For Traditionalists\n1. **Turkish coffee at Fazıl Bey** - Ultimate traditional experience\n2. **Coffee fortune telling** - Cultural entertainment\n3. **Grand Bazaar coffee break** - Historic atmosphere\n4. **Waterfront coffee in Üsküdar** - Traditional with views\n\n### For Modern Coffee Lovers\n1. **Single-origin tasting** - Explore global flavors\n2. **Barista workshops** - Learn professional techniques\n3. **Roastery visits** - See beans transform\n4. **Coffee cocktails** - Evening coffee culture\n\n### Unique Istanbul Experiences\n1. **Turkish coffee with lokum** - Traditional pairing\n2. **Bosphorus ferry coffee** - Mobile coffee culture\n3. **Rooftop coffee with city views** - Modern meeting traditional\n4. **Market coffee breaks** - Local life immersion\n\n## 💰 Coffee Economics\n\n### Price Ranges\n- **Traditional Turkish coffee**: 8-15 TL\n- **Specialty espresso drinks**: 15-35 TL\n- **Premium single-origin**: 20-45 TL\n- **Coffee with view**: 25-60 TL (tourist areas)\n\n### Value Tips\n- **Local neighborhoods**: Better prices than tourist areas\n- **Happy hours**: Some cafes offer afternoon discounts\n- **Loyalty programs**: Many independent shops offer cards\n- **Multiple visits**: Build relationships for better service\n\n## 🎯 Coffee Etiquette and Culture\n\n### Traditional Coffee Culture\n- **Patience**: Turkish coffee takes time, don't rush\n- **Conversation**: Coffee is social, engage with others\n- **Respect**: Older generation has coffee wisdom\n- **Grounds**: Don't drink the sediment at bottom\n\n### Modern Cafe Culture\n- **Wifi**: Many cafes welcome laptop users\n- **Time**: No pressure to leave quickly\n- **Tipping**: Round up bill or 10% for good service\n- **Language**: English increasingly common\n\n## 🌟 Future of Istanbul Coffee\n\n### Emerging Trends\n- **Sustainability**: Focus on ethical sourcing\n- **Local roasting**: More neighborhood roasteries\n- **Education**: Coffee appreciation courses\n- **Innovation**: Turkish flavors in modern preparation\n- **Technology**: App-based ordering, delivery services\n\n### Cultural Bridge\nIstanbul's coffee scene beautifully bridges its Ottoman heritage with global coffee culture. Whether you prefer the ritualistic preparation of traditional Turkish coffee or the precision of modern specialty brewing, Istanbul offers coffee experiences that honor the past while embracing innovation.\n\n*What's your favorite Istanbul coffee experience? Traditional Turkish coffee or modern specialty brews?*	Coffee Culture Expert	\N	2025-09-23 22:11:00	1
3	Weekend Markets of Istanbul: A Local's Shopping Guide	# Discover Istanbul's Vibrant Weekend Market Scene\n\nIstanbul's weekend markets offer an authentic glimpse into local life, where residents shop for fresh produce, vintage treasures, and unique crafts. These bustling bazaars are where the real Istanbul comes alive, away from tourist crowds.\n\n## 🥕 Fresh Produce Markets\n\n### Kadıköy Tuesday Market\nThe city's most beloved farmers market:\n- **When**: Every Tuesday, 8 AM - 6 PM\n- **Location**: Central Kadıköy, Asian side\n- **Specialty**: Organic vegetables, artisanal cheeses\n- **Atmosphere**: Local families, elderly shoppers, quality focus\n- **Best buys**: Seasonal fruits, Turkish pickles, fresh herbs\n- **Pro tip**: Arrive early for best selection\n\n### Beşiktaş Saturday Market\nUpscale neighborhood's organic focus:\n- **Schedule**: Saturdays, 9 AM - 5 PM\n- **Character**: Health-conscious, premium quality\n- **Vendors**: Local farmers, organic producers\n- **Prices**: Higher than average but exceptional quality\n- **Specialties**: Heirloom tomatoes, artisanal bread, raw honey\n- **Crowd**: Young professionals, health enthusiasts\n\n### Fatih Wednesday Market\nTraditional neighborhood market:\n- **Location**: Historic Fatih district\n- **Atmosphere**: Conservative, family-oriented\n- **Products**: Traditional Turkish vegetables, spices\n- **Prices**: Very affordable, local pricing\n- **Language**: Primarily Turkish spoken\n- **Experience**: Most authentic local market experience\n\n## 🎨 Artisan and Craft Markets\n\n### Galata Weekend Art Market\nCreative hub for local artists:\n- **When**: Saturdays and Sundays, 11 AM - 7 PM\n- **Location**: Galata district streets\n- **Vendors**: Local artists, designers, craftspeople\n- **Products**: Handmade jewelry, original artwork, ceramics\n- **Atmosphere**: Bohemian, artistic, international\n- **Prices**: Mid-range, supporting local artists\n\n### Arnavutköy Crafts Market\nBosphorus village charm:\n- **Schedule**: Sunday mornings, seasonal\n- **Setting**: Historic wooden houses backdrop\n- **Specialties**: Traditional Turkish crafts, textiles\n- **Vendors**: Local artisans, family businesses\n- **Products**: Hand-woven textiles, traditional pottery\n- **Views**: Bosphorus waterfront while shopping\n\n## 👗 Vintage and Second-Hand Markets\n\n### Beyoğlu Flea Market\nTreasure hunting in historic district:\n- **Location**: Various streets around İstiklal Avenue\n- **When**: Saturday and Sunday afternoons\n- **Finds**: Vintage clothing, old books, antique items\n- **Sellers**: Collectors, vintage enthusiasts, students\n- **Bargaining**: Essential skill, expect to negotiate\n- **Hidden gems**: Ottoman-era items, vintage Turkish textiles\n\n### Kadıköy Vintage Market\nAsian side's retro haven:\n- **Setting**: Covered and outdoor stalls\n- **Specialties**: 1960s-80s clothing, vinyl records\n- **Crowd**: Young, alternative, music lovers\n- **Prices**: Very reasonable, student-friendly\n- **Unique finds**: Turkish pop records, vintage band t-shirts\n- **Atmosphere**: Relaxed, unhurried browsing\n\n## 🏺 Antique and Collectibles Markets\n\n### Çukurcuma Antique Street\nWeekend antique hunting:\n- **Location**: Historic Beyoğlu neighborhood\n- **Character**: Professional dealers, serious collectors\n- **Items**: Ottoman antiques, Turkish carpets, old maps\n- **Quality**: High-end, authenticated pieces\n- **Prices**: Expensive but genuine antiques\n- **Expertise**: Knowledgeable dealers with stories\n\n### Horhor Antique Market\nCovered antique bazaar:\n- **Setting**: Multi-story covered market\n- **Variety**: Furniture, jewelry, books, curiosities\n- **Atmosphere**: Maze-like, treasure hunt feeling\n- **Prices**: Range from affordable to expensive\n- **Negotiation**: Expected and part of the experience\n- **Time needed**: Several hours to explore thoroughly\n\n## 🧵 Textile and Fashion Markets\n\n### Merter Textile Market\nWholesale fashion district:\n- **When**: Weekdays and Saturday mornings\n- **Focus**: Clothing manufacturing, wholesale\n- **Prices**: Factory prices, bulk buying\n- **Quality**: Range from basic to high-end\n- **Bargaining**: Essential, expect significant discounts\n- **Best for**: Fashion enthusiasts, small business owners\n\n### Tahtakale Traditional Textiles\nHistoric textile market:\n- **Location**: Near Grand Bazaar and Spice Market\n- **Specialties**: Traditional Turkish fabrics, carpets\n- **Vendors**: Family businesses, generations of experience\n- **Products**: Silk scarves, traditional patterns, kilims\n- **Authenticity**: Genuine Turkish textiles\n- **Cultural**: Part of Ottoman trading tradition\n\n## 🌿 Specialty Markets\n\n### Balık Pazarı (Fish Market)\nGalatasaray's famous fish market:\n- **Location**: Off İstiklal Avenue, Beyoğlu\n- **When**: Daily, but weekends are most vibrant\n- **Products**: Fresh seafood, Mediterranean specialties\n- **Restaurants**: Surrounding meyhanes (taverns)\n- **Atmosphere**: Lively, social, traditional\n- **Experience**: Buy fish, have it cooked nearby\n\n### Spice Market Weekend Extensions\nSaturday spice and herb vendors:\n- **Location**: Around Eminönü Spice Bazaar\n- **Products**: Rare spices, medicinal herbs, teas\n- **Vendors**: Specialty importers, traditional healers\n- **Quality**: Premium spices not found elsewhere\n- **Knowledge**: Vendors share traditional uses\n- **Prices**: Competitive with exceptional quality\n\n## 💰 Market Shopping Strategy\n\n### Budget Planning\n- **Produce markets**: 50-100 TL for weekly shopping\n- **Artisan markets**: 100-500 TL for unique pieces\n- **Vintage markets**: 20-200 TL for clothing items\n- **Antique markets**: 100-5000+ TL for serious pieces\n\n### Bargaining Essentials\n- **Start low**: Offer 40-50% of asking price\n- **Bundle deals**: Buy multiple items for discounts\n- **Cash preferred**: Bring Turkish lira\n- **Patience**: Take time, don't show urgency\n- **Respect**: Maintain friendly, respectful attitude\n\n### Practical Tips\n- **Bring bags**: Reusable shopping bags essential\n- **Cash**: Many vendors don't accept cards\n- **Early arrival**: Best selection before crowds\n- **Comfortable shoes**: Lots of walking on uneven surfaces\n- **Language**: Basic Turkish phrases helpful\n\n## 🚌 Getting to Markets\n\n### Public Transport\n- **Metro**: M1, M2, M4 lines serve major market areas\n- **Bus**: Extensive network, use IBB mobile app\n- **Ferry**: Scenic route to Asian side markets\n- **Tram**: T1 line serves European side markets\n- **Walking**: Many markets within walking distance\n\n### Market Hopping Routes\n**Saturday Route**: Beşiktaş Market → Ferry to Kadıköy → Vintage shopping\n**Sunday Route**: Galata Art Market → Çukurcuma Antiques → Balık Pazarı\n\n## 🎯 Local Market Etiquette\n\n### Cultural Sensitivity\n- **Modest dress**: Especially in conservative neighborhoods\n- **Photography**: Ask permission before photographing vendors\n- **Tasting**: Vendors often offer samples\n- **Crowds**: Be patient during busy periods\n- **Children**: Markets are family-friendly environments\n\n### Building Relationships\n- **Regular visits**: Vendors remember good customers\n- **Small talk**: Chat about products, family, neighborhood\n- **Loyalty**: Stick with vendors who treat you well\n- **Recommendations**: Ask vendors about other good stalls\n\n## 🌟 Hidden Market Gems\n\n### Insider Secrets\n- **End of day discounts**: Produce vendors reduce prices\n- **Seasonal specialties**: Each season brings unique products\n- **Vendor meals**: Some markets have hidden food stalls\n- **Local favorites**: Ask residents for market recommendations\n- **Off-season**: Quieter markets offer better personal service\n\n### Unique Experiences\n- **Market breakfast**: Turkish breakfast at market cafes\n- **Seasonal festivals**: Markets host special events\n- **Cooking classes**: Some vendors offer informal lessons\n- **Cultural exchange**: Practice Turkish with friendly vendors\n\nIstanbul's weekend markets offer more than shopping—they provide authentic cultural immersion, connection with local communities, and the joy of discovering unique treasures. Each market has its own personality, reflecting the character of its neighborhood and the people who shop there.\n\n*Which Istanbul market sounds most appealing to you? Fresh produce, vintage treasures, or artisan crafts?*	Local Market Expert	\N	2025-09-23 22:12:00	1
4	Istanbul's Underground: Exploring Byzantine Cisterns and Hidden Tunnels	# Journey Beneath Istanbul: Ancient Underground Wonders\n\nBeneath Istanbul's bustling streets lies a hidden world of Byzantine engineering marvels, Ottoman-era tunnels, and mysterious underground chambers. This subterranean Istanbul reveals layers of history that most visitors never see.\n\n## 🏛️ Major Byzantine Cisterns\n\n### Basilica Cistern (Yerebatan Sarnıcı)\nIstanbul's most famous underground wonder:\n- **Built**: 532 AD during Emperor Justinian's reign\n- **Size**: 143 x 65 meters, holds 80,000 cubic meters\n- **Columns**: 336 marble columns, each 9 meters high\n- **Famous features**: Medusa head column bases\n- **Atmosphere**: Mystical lighting, classical music\n- **Visitor info**: Most popular, can be crowded\n\n### Binbirdirek Cistern (Thousand and One Columns)\nQuieter alternative with unique architecture:\n- **History**: Built 4th century AD, predates Basilica Cistern\n- **Structure**: 224 columns in 16 rows\n- **Atmosphere**: Less crowded, more intimate experience\n- **Events**: Occasional concerts and art exhibitions\n- **Architecture**: Different column styles, reused materials\n- **Access**: Smaller entrance, easier to miss\n\n### Theodosius Cistern\nRecently restored hidden gem:\n- **Discovery**: Only recently opened to public\n- **Size**: Smaller but beautifully preserved\n- **Features**: Original Byzantine marble work\n- **Crowds**: Least known, most peaceful\n- **Architecture**: Excellent example of Byzantine engineering\n- **Location**: Near Beyazıt area\n\n## 🔍 Lesser-Known Underground Sites\n\n### Şerefiye Cistern\nByzantine engineering masterpiece:\n- **Location**: Sultanahmet area\n- **Features**: Most advanced Byzantine architecture\n- **Columns**: Fewer but more ornate columns\n- **Technology**: Sophisticated water filtration system\n- **Restoration**: Recently restored with modern lighting\n- **Experience**: Multimedia presentations about Byzantine life\n\n### Nakilbent Cistern\nHidden beneath a parking lot:\n- **Discovery**: Found during construction in 1980s\n- **Access**: Limited opening hours, small groups\n- **Features**: Original Byzantine brick arches\n- **Atmosphere**: Raw, unrestored authentic feel\n- **Exploration**: Flashlight tours available\n- **Unique**: Still shows construction techniques\n\n## 🕳️ Ottoman Underground Networks\n\n### Historic Tunnel Systems\nOttoman-era underground passages:\n- **Purpose**: Military defense, palace connections\n- **Topkapı Palace tunnels**: Secret royal passages\n- **Grand Bazaar tunnels**: Merchant storage areas\n- **Galata tunnels**: Genoese and Ottoman military routes\n- **Access**: Most closed to public, some tour opportunities\n\n### Underground Storage Areas\nCommercial underground spaces:\n- **Han courtyards**: Multi-level commercial complexes\n- **Spice storage**: Traditional underground preservation\n- **Ice houses**: Ottoman-era refrigeration systems\n- **Modern use**: Some converted to restaurants, galleries\n\n## 🚇 Modern Underground Istanbul\n\n### Metro Archaeological Discoveries\nModern excavations revealing ancient history:\n- **Marmaray project**: Revealed 8,000 years of history\n- **Station displays**: Archaeological finds exhibited\n- **Ongoing discoveries**: Each new metro line reveals artifacts\n- **Public access**: Some finds visible in metro stations\n- **UNESCO protection**: Balancing development with preservation\n\n### Underground Shopping and Dining\nModern subterranean spaces:\n- **Mall complexes**: Underground shopping centers\n- **Restaurant basements**: Historic building conversions\n- **Art galleries**: Converted cistern and basement spaces\n- **Cultural venues**: Underground performance spaces\n\n## 🏗️ Engineering Marvels\n\n### Byzantine Water System\nAncient hydraulic engineering:\n- **Valens Aqueduct**: Brought water from 120km away\n- **Distribution network**: Complex system of cisterns\n- **Capacity**: Could supply 1 million people\n- **Technology**: Gravity-fed, no pumps needed\n- **Preservation**: System worked for over 1,000 years\n\n### Construction Techniques\nHow ancients built underground:\n- **Materials**: Recycled columns from earlier buildings\n- **Waterproofing**: Hydraulic mortar, still functional\n- **Support systems**: Advanced arch and dome techniques\n- **Ventilation**: Natural air circulation systems\n- **Maintenance**: Access shafts for cleaning and repair\n\n## 🎯 Exploration Tips\n\n### What to Bring\n- **Comfortable shoes**: Floors can be wet and uneven\n- **Light jacket**: Underground temperature constant 13°C\n- **Camera**: Low-light photography equipment\n- **Flashlight**: Some areas poorly lit\n- **Water**: Exploration can be thirsty work\n\n### Photography Guidelines\n- **No flash**: Damages ancient structures\n- **Tripod useful**: Long exposures for best shots\n- **Respect signs**: Some areas restrict photography\n- **Share responsibly**: Avoid revealing secret locations\n\n### Safety Considerations\n- **Stay with groups**: Easy to get disoriented\n- **Follow guides**: Local expertise invaluable\n- **Watch footing**: Wet surfaces can be slippery\n- **Respect barriers**: Closed areas are dangerous\n- **Emergency contacts**: Inform someone of exploration plans\n\n## 🕐 Best Times to Visit\n\n### Avoiding Crowds\n- **Early morning**: First opening hours\n- **Weekdays**: Fewer tourists than weekends\n- **Off-season**: Winter months less crowded\n- **Late afternoon**: Tour groups have usually left\n\n### Special Experiences\n- **Night tours**: Some cisterns offer evening visits\n- **Concert events**: Classical music in cisterns\n- **Private tours**: Small group specialized experiences\n- **Photography tours**: Professional guidance for best shots\n\n## 💰 Costs and Access\n\n### Entrance Fees\n- **Basilica Cistern**: 30 TL (most expensive)\n- **Binbirdirek**: 20 TL (good value)\n- **Theodosius**: 15 TL (newest, least crowded)\n- **Combined tickets**: Sometimes available for multiple sites\n- **Student discounts**: With valid international student ID\n\n### Guided Tours\n- **Professional guides**: 200-400 TL for private groups\n- **Audio guides**: Available at major cisterns\n- **Specialized tours**: Underground photography, engineering focus\n- **Group rates**: Discounts for 10+ people\n\n## 🗺️ Underground Walking Route\n\n### Half-Day Underground Tour\n1. **Start**: Basilica Cistern (most famous)\n2. **Walk**: Binbirdirek Cistern (15 minutes)\n3. **Lunch**: Traditional restaurant with basement dining\n4. **Explore**: Theodosius Cistern\n5. **End**: Underground shopping in historic han\n\n### Full-Day Underground Adventure\n- **Morning**: Major cisterns tour\n- **Afternoon**: Metro archaeological stations\n- **Evening**: Underground restaurant dinner\n- **Transport**: Use metro to see underground infrastructure\n\n## 🔬 Archaeological Significance\n\n### Historical Layers\n- **Byzantine period**: 4th-15th centuries\n- **Ottoman additions**: 15th-20th centuries\n- **Modern discoveries**: Ongoing archaeological work\n- **Prehistoric findings**: Some areas show ancient settlement\n- **Cultural continuity**: How each era built upon previous\n\n### Research Opportunities\n- **University programs**: Some offer underground archaeology\n- **Volunteer excavations**: Opportunities for visitors\n- **Academic tours**: In-depth historical context\n- **Research access**: Special permits for serious study\n\n## 🌟 Hidden Underground Gems\n\n### Secret Locations\n*Note: These require local guides or special access*\n- **Private cisterns**: Under some historic buildings\n- **Tunnel networks**: Ottoman-era secret passages\n- **Underground han**: Historic commercial spaces\n- **Monastery ruins**: Byzantine religious sites\n\n### Urban Legends and Mysteries\n- **Lost tunnels**: Rumored palace connections\n- **Treasure chambers**: Ottoman-era storage rooms\n- **Escape routes**: Secret passages from palaces\n- **Unexplored areas**: Sealed sections awaiting discovery\n\n## 🎭 Cultural Experiences\n\n### Events in Underground Spaces\n- **Classical concerts**: Acoustic perfection in cisterns\n- **Art exhibitions**: Unique gallery spaces\n- **Wine tastings**: Historic atmosphere\n- **Cultural performances**: Traditional music and dance\n\n### Literary and Cinematic History\n- **Dan Brown**: Featured in Inferno novel\n- **Film locations**: Various movies shot in cisterns\n- **Literary inspiration**: Byzantine and Ottoman tales\n- **Photography**: Instagram-worthy mysterious atmosphere\n\nIstanbul's underground reveals the city's incredible depth—literally and historically. Each subterranean space tells stories of ancient engineering, religious devotion, and practical urban planning that supported millions of residents across centuries.\n\n*Have you explored Istanbul's underground? Which hidden chamber would you most like to discover?*	Underground Explorer	\N	2025-09-23 22:13:00	5
5	Bosphorus Village Life: Discovering Arnavutköy and Bebek	# Village Charm Along the Bosphorus\n\nWhile central Istanbul pulses with urban energy, the Bosphorus villages offer a different rhythm of life. These historic waterfront neighborhoods preserve Ottoman elegance while embracing modern Istanbul culture.\n\n## 🏘️ Arnavutköy: Ottoman Wooden Houses\n\n### Historic Character\nAlbanian Village charm:\n- **Architecture**: 19th-century Ottoman wooden houses\n- **Waterfront**: Traditional fishing boats, seafood restaurants\n- **Community**: Strong neighborhood bonds, local festivals\n- **Preservation**: Careful restoration of historic buildings\n- **Atmosphere**: Village life within the metropolis\n\n### Best Experiences\n- **Sunset dining**: Waterfront fish restaurants\n- **Historic walks**: Ottoman-era streets and houses\n- **Photography**: Colorful wooden houses against Bosphorus\n- **Community events**: Local festivals and celebrations\n\n## 🌊 Bebek: Upscale Waterfront Living\n\n### Sophisticated Village\nWhere elegance meets water:\n- **Setting**: Crescent-shaped bay with park\n- **Dining**: High-end restaurants, chic cafes\n- **Culture**: Art galleries, boutique shopping\n- **Recreation**: Jogging path, waterfront promenade\n- **Social**: Popular weekend destination\n\n### Signature Spots\n- **Bebek Park**: Waterfront relaxation\n- **Historic mansions**: Ottoman and Art Nouveau architecture\n- **Cafe culture**: Sophisticated coffee and brunch scene\n- **Evening atmosphere**: Romantic waterfront dining\n\nThese Bosphorus villages offer the perfect blend of history, natural beauty, and sophisticated urban culture, providing peaceful escapes without leaving the city.	Bosphorus Explorer	\N	2025-09-23 22:14:00	1
6	Istanbul's Green Spaces: Parks and Gardens for Every Season	# Urban Oases: Istanbul's Beautiful Parks and Gardens\n\nAmidst the urban intensity, Istanbul offers numerous green spaces where residents and visitors can reconnect with nature. Each park has its own character, from imperial Ottoman gardens to modern urban design.\n\n## 🌸 Historic Imperial Parks\n\n### Gülhane Park\nFormer imperial garden:\n- **History**: Part of Topkapı Palace grounds\n- **Features**: Historic trees, rose gardens, museums\n- **Views**: Golden Horn and Bosphorus glimpses\n- **Seasons**: Spring tulips, autumn colors\n- **Culture**: Outdoor concerts, art exhibitions\n\n### Yıldız Park\nSultan's private retreat:\n- **Size**: 160 hectares of wooded hills\n- **Architecture**: Ottoman pavilions and kiosks\n- **Activities**: Hiking trails, picnic areas\n- **Wildlife**: Native bird species, ancient trees\n- **Tranquility**: Peaceful escape from city noise\n\n## 🌳 Modern Urban Parks\n\n### Maçka Democracy Park\nCentral green corridor:\n- **Location**: Between Nişantaşı and Beşiktaş\n- **Features**: Cable car, jogging paths\n- **Recreation**: Outdoor gym, children's playground\n- **Events**: Weekend markets, festivals\n- **Design**: Modern landscape architecture\n\n### Fenerbahçe Park\nAsian side waterfront:\n- **Setting**: Marmara Sea coastline\n- **Activities**: Walking, cycling, sports facilities\n- **Views**: Prince Islands panorama\n- **Family**: Children's areas, picnic spots\n- **Sunset**: Popular evening destination\n\n## 🌺 Botanical Gardens and Special Collections\n\n### Nezahat Gökyiğit Botanical Garden\nTurkey's largest botanical collection:\n- **Size**: 50 hectares of diverse plant collections\n- **Education**: Plant research and conservation\n- **Seasons**: Something blooming year-round\n- **Trails**: Themed walking paths\n- **Photography**: Outstanding natural beauty\n\n### Emirgan Park\nTulip festival headquarters:\n- **Fame**: Istanbul Tulip Festival venue\n- **Trees**: Century-old plane trees\n- **Hills**: Multiple levels with different views\n- **Pavilions**: Historic Ottoman kiosks\n- **Spring**: Peak bloom season spectacular\n\nIstanbul's parks offer seasonal beauty, recreational opportunities, and peaceful retreats, proving that this urban giant maintains strong connections to the natural world.	Urban Nature Guide	\N	2025-09-23 22:15:00	2
7	Night Markets and Evening Food Culture	# After Dark: Istanbul's Vibrant Night Food Scene\n\nWhen the sun sets, Istanbul's food culture takes on new energy. Night markets, street vendors, and late-night eateries create a completely different culinary landscape worth exploring.\n\n## 🌙 Famous Night Markets\n\n### Kadıköy Tuesday Night Market\nAsian side's evening food paradise:\n- **Time**: Tuesday evenings until 11 PM\n- **Specialties**: Fresh produce, prepared foods\n- **Atmosphere**: Local families, relaxed shopping\n- **Food**: Traditional Turkish street food\n- **Community**: Neighborhood social center\n\n### Beşiktaş Saturday Evening Bazaar\nUpscale night shopping:\n- **Duration**: Saturday 6 PM - 10 PM\n- **Quality**: Premium organic products\n- **Prepared foods**: Artisanal cheese, bread, olives\n- **Wine**: Local wine tasting stalls\n- **Crowd**: Young professionals, food enthusiasts\n\n## 🍢 Street Food After Dark\n\n### Classic Night Snacks\nTraditional evening treats:\n- **Balık Ekmek**: Fresh fish sandwiches by the water\n- **Döner**: Late-night döner stands throughout city\n- **Kokoreç**: Grilled offal sandwich, acquired taste\n- **Çiğ Köfte**: Spicy bulgur balls, vegetarian option\n- **Midye**: Stuffed mussels with lemon\n\n### Modern Street Food\nContemporary late-night options:\n- **Gourmet burgers**: Artisanal burger trucks\n- **Asian fusion**: Korean-Turkish fusion stalls\n- **Craft beer**: Local brewery food trucks\n- **International**: Mexican, Lebanese, Italian options\n- **Healthy options**: Fresh juice bars, salad stands\n\n## 🏮 Night Market Districts\n\n### Beyoğlu Evening Scene\nBohemian night food culture:\n- **İstiklal Avenue**: International street food\n- **Nevizade**: Meyhane culture, shared plates\n- **Galata**: Artisanal food shops open late\n- **Karaköy**: Modern food halls, wine bars\n\n### Kadıköy After Hours\nAuthentic local night culture:\n- **Moda**: Seaside dining, fish restaurants\n- **Bahariye**: Shopping street food vendors\n- **Caferağa**: Local neighborhood eateries\n- **Ferry terminal**: Late-night commuter food\n\nIstanbul's night food scene offers everything from traditional Turkish comfort food to innovative international cuisine, creating memorable evening experiences for every taste.	Night Food Explorer	\N	2025-09-23 22:16:00	3
8	Ferry Routes and Bosphorus Transportation Guide	# Navigating Istanbul by Water: Complete Ferry Guide\n\nIstanbul's ferry system offers both practical transportation and scenic journeys. Understanding the routes, schedules, and insider tips transforms ferry travel from simple transport into memorable experiences.\n\n## ⛴️ Major Ferry Routes\n\n### Eminönü - Üsküdar Line\nMost popular cross-Bosphorus route:\n- **Duration**: 15 minutes each way\n- **Frequency**: Every 10-15 minutes during peak hours\n- **Views**: Old City, Topkapı Palace, Asian side\n- **Cost**: Very affordable with İstanbulkart\n- **Tips**: Best for quick city crossing\n\n### Beşiktaş - Üsküdar Route\nNorthern Bosphorus crossing:\n- **Journey**: 20 minutes with Bosphorus views\n- **Sights**: Dolmabahçe Palace, modern skyline\n- **Commute**: Popular with daily commuters\n- **Schedule**: Frequent service throughout day\n- **Experience**: Less touristy, more authentic\n\n## 🏝️ Island Ferry Adventures\n\n### Prince Islands Route\nWeekend escape from Kabataş:\n- **Destinations**: Büyükada, Heybeliada, Burgazada\n- **Journey time**: 45 minutes to 1.5 hours\n- **Seasons**: Year-round service, best spring-fall\n- **Activities**: Island exploration, cycling, swimming\n- **Planning**: Full day trip recommended\n\n### Golden Horn Ferries\nHistoric waterway exploration:\n- **Route**: Eminönü to Eyüp and beyond\n- **History**: Traditional Ottoman waterway\n- **Sights**: Historic neighborhoods, mosques\n- **Culture**: Local commuter experience\n- **Photography**: Urban and historic contrasts\n\n## 🚢 Bosphorus Cruise Options\n\n### Short Bosphorus Tours\nQuick scenic journeys:\n- **Duration**: 1.5 - 2 hours round trip\n- **Departure**: Eminönü, Beşiktaş, Kabataş\n- **Highlights**: Bosphorus Bridge, waterfront palaces\n- **Frequency**: Multiple departures daily\n- **Cost**: Budget-friendly sightseeing option\n\n### Full Bosphorus Experience\nComplete strait exploration:\n- **Length**: 6-8 hours with stops\n- **Destination**: Black Sea entrance and return\n- **Stops**: Anadolu Kavağı for lunch and exploration\n- **Views**: Both European and Asian coastlines\n- **Experience**: Full day adventure\n\n## 💡 Ferry Travel Tips\n\n### Best Times to Travel\n- **Sunrise**: Magical golden hour lighting\n- **Sunset**: Spectacular evening colors\n- **Midweek**: Less crowded than weekends\n- **Off-season**: Calmer waters, clear views\n\n### What to Bring\n- **İstanbulkart**: Essential for all public transport\n- **Light jacket**: Can be windy on water\n- **Camera**: Outstanding photography opportunities\n- **Snacks**: Tea and simit available on board\n- **Patience**: Schedules can vary with weather\n\n## 🎯 Insider Ferry Secrets\n\n### Local Commuter Culture\n- **Tea service**: Traditional Turkish tea on board\n- **Seagull feeding**: Bring bread for the birds\n- **Deck preference**: Upper deck for views, lower for warmth\n- **Rush hours**: Experience authentic Istanbul commute\n- **Weather watching**: Service affected by storms\n\n### Photography Tips\n- **Golden hour**: 1 hour before/after sunrise/sunset\n- **Positioning**: Bow and stern offer best angles\n- **Equipment**: Steady shots challenging in motion\n- **Subjects**: Architecture, daily life, seascapes\n- **Etiquette**: Respectful of other passengers\n\nFerry travel in Istanbul offers unmatched views, cultural immersion, and practical transportation, making it essential for understanding the city's maritime character and daily rhythms.	Maritime Istanbul Guide	\N	2025-09-23 22:17:00	1
9	Traditional Turkish Breakfast: A Cultural Institution	# Kahvaltı: The Art of Turkish Breakfast\n\nTurkish breakfast is not just a meal—it's a cultural institution that brings families together and sets the pace for the day. Understanding this tradition opens doors to authentic Istanbul experiences.\n\n## 🥖 Essential Components\n\n### The Foundation Elements\nEvery Turkish breakfast includes:\n- **Fresh bread**: Crusty white bread, simit (Turkish bagel)\n- **Cheese varieties**: White cheese, kashkaval, tulum cheese\n- **Olives**: Green and black varieties, marinated options\n- **Tomatoes and cucumbers**: Fresh, seasonal vegetables\n- **Honey and jam**: Local honey, rose petal jam, fig preserve\n- **Turkish tea**: Strong black tea in tulip glasses\n\n### Regional Specialties\nIstanbul additions:\n- **Kaymak**: Clotted cream, rich and indulgent\n- **Bal-kaymak**: Honey with clotted cream combination\n- **Pastries**: Börek, su böreği, cheese pastries\n- **Eggs**: Various preparations, often with pastırma\n- **Seasonal fruits**: Whatever's fresh and local\n\n## 🏨 Best Breakfast Experiences\n\n### Traditional Family Places\nAuthentic neighborhood spots:\n- **Çukurcuma Köftecisi**: Historic neighborhood breakfast\n- **Pandeli**: Ottoman-era restaurant near Spice Bazaar\n- **Hamdi Restaurant**: Famous for traditional spreads\n- **Local neighborhood lokanta**: Authentic daily breakfast\n\n### Modern Interpretations\nContemporary Turkish breakfast:\n- **Karakoy Lokantasi**: Upscale traditional breakfast\n- **House Cafe**: Modern twist on classics\n- **Mentha**: Health-conscious organic options\n- **Nopa**: International influences with Turkish base\n\nTurkish breakfast represents community, family time, and the Turkish philosophy that good food should be shared and savored, never rushed.	Breakfast Culture Expert	\N	2025-09-23 22:18:00	4
10	Istanbul's Music Scene: From Classical to Contemporary	# The Sounds of Istanbul: A Musical Journey\n\nIstanbul's musical landscape reflects its cultural diversity, with venues ranging from classical Ottoman performances to cutting-edge electronic music clubs.\n\n## 🎼 Classical and Traditional Music\n\n### Ottoman Classical Music\nHistoric musical traditions:\n- **Venues**: Hodjapasha Cultural Center, Süreyya Opera House\n- **Instruments**: Oud, ney, kanun, percussion\n- **Performances**: Traditional fasıl evenings\n- **Learning**: Workshops and masterclasses available\n- **Cultural context**: Understanding Ottoman court music\n\n### Turkish Folk Music\nAnatolian musical heritage:\n- **Styles**: Regional folk traditions from across Turkey\n- **Instruments**: Bağlama, zurna, davul\n- **Venues**: Cultural centers, traditional restaurants\n- **Festivals**: Annual folk music celebrations\n- **Dance**: Traditional folk dancing accompaniment\n\n## 🎸 Contemporary Music Scene\n\n### Rock and Alternative\nModern Turkish rock:\n- **Legendary bands**: Barış Manço legacy, Duman, Şebnem Ferah\n- **Venues**: Babylon, IF Performance Hall, Zorlu PSM\n- **Scene**: Active local bands, international acts\n- **Underground**: Independent venues in Kadıköy\n- **Language**: Mix of Turkish and English performances\n\n### Electronic Music\nClub culture and electronic scene:\n- **Clubs**: Suma Beach, Klein Phönix, Blackk\n- **DJs**: Local and international electronic artists\n- **Genres**: Techno, house, experimental electronic\n- **Festivals**: Annual electronic music events\n- **Community**: Dedicated electronic music following\n\nIstanbul's music scene bridges centuries of tradition with contemporary innovation, offering something for every musical taste.	Music Scene Insider	\N	2025-09-23 22:19:00	2
11	Shopping Districts: From Grand Bazaar to Modern Malls	# Istanbul Shopping Guide: Traditional to Contemporary\n\nIstanbul offers shopping experiences spanning centuries, from ancient bazaars to ultramodern shopping centers, each reflecting different aspects of the city's commercial culture.\n\n## 🏪 Historic Bazaars\n\n### Grand Bazaar (Kapalıçarşı)\nWorld's oldest covered market:\n- **History**: Built 1461, 4,000 shops\n- **Structure**: 61 covered streets, medieval architecture\n- **Products**: Carpets, jewelry, ceramics, textiles\n- **Negotiation**: Bargaining expected and encouraged\n- **Culture**: Traditional merchant relationships\n- **Tips**: Visit early morning or late afternoon\n\n### Spice Bazaar (Mısır Çarşısı)\nAromatic trading center:\n- **Specialties**: Spices, Turkish delight, dried fruits\n- **History**: 17th-century Egyptian Bazaar\n- **Quality**: Premium spices and traditional products\n- **Samples**: Vendors offer tastings\n- **Gifts**: Perfect for culinary souvenirs\n- **Atmosphere**: Sensory overload of colors and scents\n\n## 🛍️ Modern Shopping Areas\n\n### İstiklal Avenue\nPedestrian shopping street:\n- **Length**: 1.4 km of shops and entertainment\n- **Variety**: International brands, local boutiques\n- **Historic passages**: Covered shopping arcades\n- **Culture**: Street performers, cultural events\n- **Dining**: Restaurants and cafes throughout\n- **Transport**: Historic tram, easy metro access\n\n### Nişantaşı District\nUpscale shopping neighborhood:\n- **Character**: High-end boutiques, luxury brands\n- **International**: Designer flagship stores\n- **Local**: Turkish fashion designers\n- **Dining**: Sophisticated restaurants and cafes\n- **Atmosphere**: European-style shopping district\n- **Services**: Personal shopping, styling services\n\n## 🏬 Shopping Malls\n\n### İstinye Park\nPremium shopping destination:\n- **Brands**: International luxury and mid-range\n- **Design**: Open-air and covered sections\n- **Dining**: Food court and fine dining options\n- **Entertainment**: Cinema, cultural events\n- **Location**: Easy access from city center\n- **Parking**: Ample parking facilities\n\n### Cevahir Mall\nLarge-scale shopping center:\n- **Size**: One of Europe's largest malls\n- **Variety**: Hundreds of international and local brands\n- **Entertainment**: Roller coaster, cinema complex\n- **Food**: Extensive food court options\n- **Services**: All shopping needs under one roof\n- **Accessibility**: Metro connection, family facilities\n\nIstanbul's shopping landscape offers everything from traditional handicrafts to international luxury, making it a paradise for shoppers seeking both authenticity and modernity.	Shopping Expert	\N	2025-09-23 22:20:00	3
12	Turkish Bath Experience: Traditional Hammam Guide	# The Art of Turkish Hammam: Complete Experience Guide\n\nThe Turkish bath, or hammam, represents centuries of bathing culture that combines cleansing, relaxation, and social tradition. Understanding hammam etiquette ensures an authentic and comfortable experience.\n\n## 🛁 Traditional Hammam Process\n\n### The Experience Steps\nClassic hammam journey:\n1. **Changing room**: Remove clothes, wear peştemal (cotton wrap)\n2. **Warm room**: Gradual temperature adjustment\n3. **Hot room**: Perspiration and preparation\n4. **Marble slab**: Central heated marble platform (göbektaşı)\n5. **Scrubbing**: Attendant uses rough mitt (kese)\n6. **Soap massage**: Foam massage with olive oil soap\n7. **Rinse**: Warm water cleansing\n8. **Cool down**: Gradual temperature reduction\n9. **Rest**: Tea and relaxation in cooling area\n\n### Health Benefits\nPhysical and mental wellness:\n- **Circulation**: Heat improves blood flow\n- **Skin**: Deep exfoliation and cleansing\n- **Muscles**: Heat therapy for tension relief\n- **Detox**: Sweating eliminates toxins\n- **Stress relief**: Meditative, calming experience\n- **Sleep**: Deep relaxation promotes better rest\n\n## 🏛️ Historic Hammams\n\n### Cağaloğlu Hammam\nIstanbul's oldest public bath:\n- **Built**: 1741, continuous operation since\n- **Architecture**: Classical Ottoman design\n- **Sections**: Separate areas for men and women\n- **Services**: Traditional bath, massage, scrub\n- **Atmosphere**: Authentic historical setting\n- **Famous visitors**: Kings, celebrities, writers\n\n### Galatasaray Hammam\nHistoric Beyoğlu location:\n- **Heritage**: 500+ years of bathing tradition\n- **Architecture**: Restored Ottoman structure\n- **Services**: Full traditional hammam experience\n- **Location**: Central Beyoğlu, easy access\n- **Quality**: Professional attendants, authentic methods\n- **Atmosphere**: Less touristy, more local experience\n\n## 💡 Hammam Etiquette and Tips\n\n### What to Expect\n- **Nudity**: Minimal clothing, peştemal covering\n- **Attendants**: Professional tellak (male) or natır (female)\n- **Language**: Basic English usually available\n- **Duration**: 1-2 hours total experience\n- **Intensity**: Scrubbing can be vigorous\n- **Privacy**: Single-sex environments\n\n### Practical Preparation\n- **Booking**: Advance reservation recommended\n- **Timing**: Allow 2-3 hours total\n- **Health**: Avoid if pregnant, heart conditions\n- **Hydration**: Drink water before and after\n- **Expectations**: Cultural experience, not luxury spa\n- **Respect**: Follow staff instructions, cultural norms\n\nThe hammam experience connects visitors to centuries of Turkish bathing culture while providing genuine wellness benefits and cultural understanding.	Traditional Culture Guide	\N	2025-09-23 22:21:00	5
13	Seasonal Istanbul: What to Expect Throughout the Year	# Istanbul Through the Seasons: Year-Round Travel Guide\n\nIstanbul's continental climate creates distinct seasonal experiences, each offering unique advantages for visitors. Understanding seasonal patterns helps optimize your Istanbul adventure.\n\n## 🌸 Spring (March - May)\n\n### Weather and Atmosphere\n- **Temperature**: 15-20°C, mild and comfortable\n- **Rainfall**: Moderate, occasional spring showers\n- **Crowds**: Increasing tourism, still manageable\n- **Nature**: Tulip season, blooming parks\n- **Daylight**: Longer days, pleasant evening walks\n\n### Spring Highlights\n- **Tulip Festival**: Emirgan Park spectacular displays\n- **Outdoor dining**: Restaurant terraces reopen\n- **Park activities**: Perfect weather for green spaces\n- **Walking tours**: Comfortable temperatures for exploration\n- **Photography**: Clear air, beautiful natural light\n\n## ☀️ Summer (June - August)\n\n### Hot Season Characteristics\n- **Temperature**: 25-30°C, hot and humid\n- **Tourism**: Peak season, crowded attractions\n- **Activities**: Beach trips, Bosphorus cruises\n- **Festivals**: Outdoor concerts, cultural events\n- **Challenges**: Heat, crowds, higher prices\n\n### Summer Strategies\n- **Early mornings**: Visit attractions before crowds\n- **Shade seeking**: Covered markets, museums\n- **Water activities**: Ferry rides, seaside dining\n- **Evening exploration**: Cooler temperatures after sunset\n- **Hydration**: Constant water consumption essential\n\n## 🍂 Autumn (September - November)\n\n### Ideal Visiting Season\n- **Weather**: 15-25°C, perfect temperatures\n- **Crowds**: Fewer tourists, more authentic experiences\n- **Colors**: Beautiful autumn foliage\n- **Activities**: All outdoor activities comfortable\n- **Festivals**: Cultural events, harvest celebrations\n\n### Autumn Advantages\n- **Photography**: Golden light, colorful landscapes\n- **Walking**: Perfect weather for exploration\n- **Local life**: Schools return, normal city rhythm\n- **Prices**: Lower accommodation rates\n- **Food**: Seasonal produce, comfort foods\n\n## ❄️ Winter (December - February)\n\n### Cold Season Experience\n- **Temperature**: 5-10°C, cold and wet\n- **Tourism**: Lowest crowds, authentic local life\n- **Activities**: Indoor attractions, cozy cafes\n- **Atmosphere**: Moody, dramatic city views\n- **Challenges**: Rain, limited daylight\n\n### Winter Pleasures\n- **Museums**: Perfect season for indoor exploration\n- **Hammams**: Traditional baths especially appealing\n- **Local culture**: Authentic neighborhood life\n- **Cozy dining**: Traditional winter foods\n- **Solitude**: Peaceful attraction visits\n\nEach season offers distinct Istanbul experiences—spring's renewal, summer's energy, autumn's perfection, and winter's intimacy.	Seasonal Travel Expert	\N	2025-09-23 22:22:00	2
14	Local Transportation Mastery: Metro, Trams, and Buses	# Mastering Istanbul Public Transportation\n\nNavigating Istanbul's extensive public transportation system efficiently transforms your city experience. Understanding the network saves time, money, and provides authentic local interactions.\n\n## 🚇 Metro System\n\n### Major Lines\n- **M1**: Airport to city center (Yenikapi)\n- **M2**: Golden Horn crossing (Vezneciler-Haciosman)\n- **M3**: Asian side backbone (Kirazli-Başakşehir)\n- **M4**: Kadıköy to Sabiha Gökçen Airport\n- **M5**: Northern European side connection\n- **M7**: Kabataş-Mecidiyeköy connection\n\n### Metro Tips\n- **İstanbulkart**: Essential rechargeable transport card\n- **Rush hours**: 7-9 AM, 5-7 PM very crowded\n- **Air conditioning**: Relief during hot summers\n- **Safety**: Generally safe, watch belongings\n- **Accessibility**: Most stations wheelchair accessible\n\n## 🚊 Tram Network\n\n### T1 Historic Line\nMost tourist-relevant route:\n- **Route**: Kabataş to Bağcılar\n- **Stops**: Sultanahmet, Eminönü, Beyazıt, Grand Bazaar\n- **Frequency**: Every 5-10 minutes\n- **Scenic**: Views of historic areas\n- **Tourist**: Connects major attractions\n\n### Nostalgic Tram\nHistoric İstiklal Avenue:\n- **Route**: Tünel to Taksim Square\n- **Character**: Historic red trams, tourist attraction\n- **Experience**: Slow, scenic ride\n- **Photography**: Iconic Istanbul image\n- **Alternative**: Walking often faster\n\n## 🚌 Bus System\n\n### Comprehensive Network\n- **Coverage**: Reaches every neighborhood\n- **Types**: Regular buses, BRT system\n- **Payment**: İstanbulkart only\n- **Challenges**: Traffic delays, crowded\n- **Local**: Most authentic transportation experience\n\n### Metrobüs (BRT)\nRapid transit system:\n- **Route**: Dedicated lanes across city\n- **Speed**: Faster than regular buses\n- **Capacity**: High-capacity vehicles\n- **Stops**: Major districts and connections\n- **Peak times**: Very crowded during rush\n\n## 💳 Payment and Cards\n\n### İstanbulkart\nUniversal transportation card:\n- **Purchase**: Stations, kiosks, some shops\n- **Deposit**: 13 TL card fee plus credit\n- **Discounts**: Transfers within 2 hours discounted\n- **Validity**: Works on all public transport\n- **Refund**: Possible at specific locations\n\n### Mobile Payment\nModern payment options:\n- **BiP app**: Mobile ticketing option\n- **Contactless**: Some credit cards accepted\n- **QR codes**: Limited implementation\n- **Cash**: Not accepted on public transport\n- **Tourist cards**: Special visitor transport passes\n\nMastering Istanbul's public transport opens the entire city while providing authentic cultural experiences and significant cost savings.	Transportation Expert	\N	2025-09-23 22:23:00	6
15	Art Galleries and Contemporary Culture	# Istanbul's Thriving Contemporary Art Scene\n\nIstanbul has emerged as a major contemporary art hub, with world-class galleries, innovative spaces, and a vibrant artist community creating dynamic cultural experiences.\n\n## 🎨 Major Art Museums\n\n### Istanbul Modern\nTurkey's premier contemporary art museum:\n- **Collection**: Modern and contemporary Turkish art\n- **Exhibitions**: International rotating shows\n- **Location**: Galataport, waterfront setting\n- **Architecture**: Stunning modern building\n- **Programs**: Artist talks, workshops\n- **Cafe**: Bosphorus views while dining\n\n### Pera Museum\nPrivate museum excellence:\n- **Focus**: Orientalist paintings, Anatolian weights\n- **Exhibitions**: High-quality temporary shows\n- **Building**: Restored historic hotel\n- **Collection**: Ottoman-era art and artifacts\n- **Location**: Central Beyoğlu\n- **Educational**: Extensive program offerings\n\n## 🏛️ Independent Gallery Spaces\n\n### Galata District Galleries\nConcentrated art scene:\n- **Galerist**: Contemporary Turkish artists\n- **Pilevneli Gallery**: International contemporary art\n- **Dirimart**: Experimental and emerging artists\n- **Alan Istanbul**: Multimedia art spaces\n- **Walkable**: Gallery hopping in historic neighborhood\n\n### Karakoy Contemporary\nIndustrial spaces turned galleries:\n- **Character**: Converted warehouses, raw spaces\n- **Artists**: Emerging local and international talent\n- **Events**: Opening nights, artist studios\n- **Atmosphere**: Edgy, alternative culture\n- **Gentrification**: Rapidly evolving neighborhood\n\n## 🎭 Alternative Art Spaces\n\n### Artist Collectives\nCommunity-driven spaces:\n- **DEPO**: Cultural center and exhibition space\n- **5533**: Artist-run contemporary space\n- **Outlet**: Experimental art and performance\n- **Community**: Local artist networking\n- **Accessibility**: Often free or low-cost entry\n\n### Pop-up Exhibitions\nTemporary art experiences:\n- **Locations**: Vacant buildings, unusual spaces\n- **Duration**: Days to weeks\n- **Discovery**: Social media, word-of-mouth\n- **Innovation**: Cutting-edge, experimental work\n- **Adventure**: Part of exploration culture\n\n## 🌟 Art Events and Festivals\n\n### Istanbul Biennial\nInternational art event:\n- **Frequency**: Every two years\n- **Scale**: City-wide contemporary art festival\n- **Venues**: Museums, galleries, public spaces\n- **International**: Global artist participation\n- **Impact**: Major cultural tourism draw\n\n### Art Walks and Tours\nGuided cultural experiences:\n- **Gallery walks**: Curated neighborhood tours\n- **Artist studio visits**: Behind-scenes experiences\n- **Collector tours**: Private collection access\n- **Architecture focus**: Building and art integration\n- **Local guides**: Artist and curator perspectives\n\nIstanbul's contemporary art scene reflects the city's position at cultural crossroads, where traditional influences meet global contemporary practice.	Contemporary Art Guide	\N	2025-09-23 22:24:00	4
16	Neighborhood Guide: Balat and Fener Historic Districts	# Balat and Fener: Historic Multicultural Neighborhoods\n\nThese adjacent Golden Horn neighborhoods showcase Istanbul's multicultural heritage, with colorful Ottoman houses, historic religious sites, and emerging gentrification creating complex cultural landscapes.\n\n## 🏘️ Balat District Character\n\n### Historic Jewish Quarter\nCenturies of Sephardic culture:\n- **Architecture**: Colorful Ottoman houses on steep streets\n- **History**: Sephardic Jewish settlement since 1492\n- **Synagogues**: Historic Ahrida and Yanbol synagogues\n- **Restoration**: Ongoing neighborhood renewal\n- **Photography**: Instagram-famous colorful facades\n\n### Modern Transformation\nGentrification and change:\n- **Cafes**: Trendy coffee shops in restored buildings\n- **Artists**: Studios and galleries opening\n- **Tourism**: Increasing visitor interest\n- **Authenticity**: Balancing development with heritage\n- **Community**: Long-term residents adapting to change\n\n## ⛪ Fener Greek Heritage\n\n### Greek Orthodox Center\nEcumenical Patriarch headquarters:\n- **St. George Cathedral**: Patriarchal church\n- **Fener Greek School**: Red brick landmark building\n- **Community**: Remaining Greek Orthodox population\n- **Architecture**: 19th-century institutional buildings\n- **Significance**: Global Orthodox Christianity center\n\n### Cultural Preservation\nMaintaining Greek heritage:\n- **Language**: Greek still spoken by elderly residents\n- **Traditions**: Orthodox religious celebrations\n- **Architecture**: Byzantine and Ottoman influences\n- **Education**: Greek cultural programs\n- **Tourism**: Heritage trail development\n\n## 🎨 Contemporary Culture\n\n### Artistic Renaissance\nCreative community growth:\n- **Studios**: Artists attracted by affordable spaces\n- **Galleries**: Small independent exhibition spaces\n- **Events**: Neighborhood art walks\n- **Design**: Architecture and interior design shops\n- **Innovation**: Traditional crafts meeting modern design\n\n### Culinary Scene\nEmerging restaurant culture:\n- **Traditional**: Authentic neighborhood restaurants\n- **Modern**: Contemporary Turkish cuisine\n- **International**: Global food influences\n- **Atmosphere**: Intimate, local dining experiences\n- **Value**: Reasonable prices compared to tourist areas\n\n## 🚶 Exploring the Neighborhoods\n\n### Walking Route Suggestions\nOptimal neighborhood exploration:\n1. **Start**: Balat ferry station\n2. **Colorful streets**: Instagram-worthy house photography\n3. **Ahrida Synagogue**: Historic religious site\n4. **Fener walk**: Up hill to Greek school\n5. **St. George Cathedral**: Orthodox patriarchate\n6. **Golden Horn views**: Waterfront panoramas\n7. **Local cafes**: Coffee break in restored building\n\n### Practical Tips\n- **Transportation**: Ferry from Eminönü most scenic\n- **Timing**: Morning light best for photography\n- **Respect**: Residential area, be considerate\n- **Comfort**: Steep streets, wear walking shoes\n- **Language**: Turkish helpful, some English available\n\nBalat and Fener offer authentic experiences of Istanbul's multicultural past while showcasing contemporary urban transformation.	Neighborhood Expert	\N	2025-09-23 22:25:00	7
17	Food Markets and Culinary Adventures	# Istanbul's Food Markets: Culinary Treasure Hunting\n\nIstanbul's food markets offer immersive culinary experiences where traditional ingredients, regional specialties, and local food culture create authentic gastronomic adventures.\n\n## 🥬 Traditional Markets\n\n### Kadıköy Tuesday Market\nAsian side's premier food market:\n- **Schedule**: Every Tuesday, early morning to evening\n- **Character**: Local families, authentic atmosphere\n- **Products**: Seasonal vegetables, Aegean olives, artisanal cheese\n- **Prices**: Local pricing, excellent value\n- **Experience**: Tasting encouraged, relationship-building\n- **Transportation**: Easy ferry access from European side\n\n### Beşiktaş Saturday Market\nUpscale organic focus:\n- **Quality**: Premium organic produce\n- **Vendors**: Small-scale sustainable farmers\n- **Products**: Heirloom vegetables, artisanal bread\n- **Atmosphere**: Health-conscious, educated clientele\n- **Prices**: Higher but exceptional quality\n- **Education**: Vendors knowledgeable about growing methods\n\n## 🐟 Specialized Food Markets\n\n### Kumkapı Fish Market\nSeafood lover's paradise:\n- **Location**: Historic fishing district\n- **Products**: Daily fresh catch, Marmara and Black Sea fish\n- **Restaurants**: Surrounding tavernas cook your purchases\n- **Atmosphere**: Traditional fishing culture\n- **Timing**: Early morning for best selection\n- **Expertise**: Vendors help with fish selection and preparation\n\n### Spice Bazaar Extensions\nAromatic ingredient hunting:\n- **Beyond tourism**: Vendors selling to locals\n- **Specialty shops**: Rare spices, medicinal herbs\n- **Quality**: Professional chef suppliers\n- **Knowledge**: Traditional uses and recipes\n- **Bulk buying**: Better prices for larger quantities\n\n## 🧀 Artisanal Food Producers\n\n### Cheese Specialists\nTurkish dairy excellence:\n- **Varieties**: Regional cheeses from across Anatolia\n- **Tasting**: Samples of different aging processes\n- **Education**: Traditional production methods\n- **Pairings**: Suggested accompaniments and wines\n- **Storage**: Proper handling and preservation advice\n\n### Olive Oil Producers\nLiquid gold varieties:\n- **Origins**: Single-estate Aegean and Mediterranean oils\n- **Tastings**: Understanding flavor profiles\n- **Harvest info**: Seasonal variations and processing\n- **Uses**: Culinary applications and health benefits\n- **Quality**: Award-winning small producers\n\n## 🍞 Bakery Culture\n\n### Neighborhood Bakeries\nDaily bread traditions:\n- **Fresh bread**: Multiple daily bakings\n- **Traditional**: Recipes passed through generations\n- **Variety**: Different regional bread styles\n- **Community**: Social center of neighborhoods\n- **Timing**: Best selection at specific hours\n\n### Specialty Pastries\nTurkish sweet traditions:\n- **Baklava**: Fresh daily preparation\n- **Regional sweets**: Specialties from different Turkish regions\n- **Seasonal**: Holiday and celebration pastries\n- **Techniques**: Traditional preparation methods\n- **Freshness**: Made-to-order options\n\n## 🍯 Market Shopping Tips\n\n### Best Practices\n- **Early arrival**: Best selection and cooler temperatures\n- **Cash preferred**: Small vendors may not accept cards\n- **Bags**: Bring reusable shopping bags\n- **Tasting**: Don't hesitate to try before buying\n- **Relationships**: Building connections with vendors\n\n### Cultural Etiquette\n- **Patience**: Take time for conversations\n- **Respect**: Appreciate vendor expertise\n- **Bargaining**: Appropriate for large purchases\n- **Language**: Basic Turkish phrases appreciated\n- **Photography**: Ask permission before photographing vendors\n\nIstanbul's food markets offer direct connections to Turkey's agricultural heritage and culinary traditions while supporting local producers and communities.	Culinary Market Expert	\N	2025-09-23 22:26:00	3
18	Hidden Architectural Gems Throughout the City	# Architectural Treasures: Beyond the Famous Monuments\n\nWhile Hagia Sophia and Blue Mosque draw millions, Istanbul harbors countless architectural gems that reveal the city's diverse building traditions and innovative design history.\n\n## 🏛️ Byzantine Hidden Treasures\n\n### Chora Church (Kariye Museum)\nByzantine mosaic masterpiece:\n- **Mosaics**: 14th-century artistic pinnacle\n- **Location**: Edirnekapı, away from tourist crowds\n- **Preservation**: Extraordinary detail and colors\n- **History**: Monastery church converted to museum\n- **Art**: Biblical scenes in stunning detail\n- **Atmosphere**: Intimate space, spiritual ambiance\n\n### Little Hagia Sophia\nJustinian's architectural prototype:\n- **Official name**: Church of Saints Sergius and Bacchus\n- **Architecture**: Practice run for Hagia Sophia\n- **Conversion**: Ottoman mosque with original structure\n- **Details**: Carved capitals, inscriptions\n- **Scale**: Human-sized, accessible experience\n- **Neighborhood**: Authentic local area\n\n## 🕌 Ottoman Architectural Innovation\n\n### Rüstem Pasha Mosque\nİznik tile masterwork:\n- **Tiles**: Extraordinary İznik ceramic interior\n- **Location**: Above Grand Bazaar shops\n- **Design**: Sinan's innovative small-space solution\n- **Colors**: Brilliant blues, greens, reds\n- **Hidden**: Easy to miss entrance\n- **Craftsmanship**: Pinnacle of Ottoman decorative arts\n\n### Mihrimah Sultan Mosques\nSinan's engineering brilliance:\n- **Two locations**: Edirnekapı and Üsküdar\n- **Innovation**: Light manipulation, structural solutions\n- **Imperial**: Sultan's daughter commissioned\n- **Views**: Strategic placement for city panoramas\n- **Engineering**: Advanced earthquake resistance\n- **Beauty**: Elegant proportions, natural lighting\n\n## 🏢 19th Century Architectural Diversity\n\n### Dolmabahçe Palace\nEuropean-influenced Ottoman grandeur:\n- **Style**: Baroque, Rococo, Neoclassical blend\n- **Scale**: 285 rooms, unprecedented luxury\n- **Technology**: Modern innovations of the era\n- **Gardens**: Elaborate landscaping\n- **Symbolism**: Ottoman modernization efforts\n- **Craftsmanship**: European and Ottoman artisans\n\n### Beyoğlu Building Heritage\nCosmopolitan architectural mix:\n- **Art Nouveau**: Early 20th-century apartment buildings\n- **Commercial**: Historic passages and shopping arcades\n- **Consulates**: Neo-classical diplomatic buildings\n- **Hotels**: Historic luxury accommodation architecture\n- **Variety**: Multiple European architectural influences\n\n## 🏗️ Modern Architectural Landmarks\n\n### Atatürk Cultural Center\nContemporary cultural architecture:\n- **Design**: Modern interpretation of Turkish motifs\n- **Function**: Opera house, concert halls, galleries\n- **Technology**: State-of-the-art acoustics and lighting\n- **Symbolism**: Modern Turkey's cultural aspirations\n- **Integration**: Respects historic Taksim context\n\n### Contemporary Residential\nModern living solutions:\n- **Bosphorus towers**: Luxury residential high-rises\n- **Adaptive reuse**: Historic building conversions\n- **Sustainable design**: Green building innovations\n- **Mixed use**: Commercial and residential integration\n- **Urban planning**: Neighborhood revitalization projects\n\n## 🔍 Architectural Discovery Tips\n\n### Finding Hidden Gems\n- **Walk slowly**: Look up and around corners\n- **Ask locals**: Residents know neighborhood treasures\n- **Photography**: Document unique details and features\n- **Research**: Historical building information online\n- **Guided tours**: Architecture-focused walking tours\n\n### What to Look For\n- **Details**: Decorative elements, inscriptions\n- **Materials**: Stone, marble, wood, metalwork\n- **Proportions**: How spaces feel and function\n- **Context**: Buildings' relationship to surroundings\n- **History**: Stories buildings tell about their eras\n\nIstanbul's architectural diversity reflects its position as cultural crossroads, where building traditions from multiple civilizations created unique hybrid styles.	Architecture Enthusiast	\N	2025-09-23 22:27:00	8
19	Festival Calendar: Cultural Events Throughout the Year	# Istanbul's Festival Calendar: Year-Round Cultural Celebrations\n\nIstanbul's event calendar showcases the city's diverse cultural heritage with festivals ranging from traditional religious celebrations to contemporary international arts events.\n\n## 🌷 Spring Festivals\n\n### Istanbul Tulip Festival (April)\nCity-wide botanical celebration:\n- **Duration**: Throughout April\n- **Locations**: Parks, gardens, public spaces city-wide\n- **Highlights**: Emirgan Park spectacular displays\n- **Activities**: Garden tours, photography competitions\n- **Free**: All events open to public\n- **Heritage**: Ottoman tulip cultivation tradition\n\n### Traditional Spring Celebrations\nSeasonal cultural events:\n- **Hıdırellez**: May 5-6, spring celebration\n- **Location**: Ahırkapı and waterfront areas\n- **Activities**: Picnics, folk dancing, wishes\n- **Community**: Local families, authentic atmosphere\n- **Traditions**: Fire jumping, rose water, fortune telling\n\n## ☀️ Summer Cultural Events\n\n### Istanbul Music Festival (June)\nClassical music excellence:\n- **Duration**: 3 weeks in June\n- **Venues**: Historic sites, concert halls\n- **Artists**: International orchestras, soloists\n- **Unique**: Concerts in Hagia Irene, cisterns\n- **Prestige**: One of Europe's premier music festivals\n\n### Istanbul Jazz Festival (July)\nInternational jazz celebration:\n- **History**: 30+ years of jazz excellence\n- **Artists**: World-renowned jazz musicians\n- **Venues**: Indoor and outdoor concert spaces\n- **Atmosphere**: Summer evening concerts\n- **Community**: Strong local jazz following\n\n## 🎭 Autumn Arts Festivals\n\n### Istanbul Biennial (Odd Years)\nContemporary art showcase:\n- **Frequency**: Every two years\n- **Duration**: September - November\n- **Scale**: City-wide contemporary art exhibition\n- **Venues**: Museums, galleries, public spaces\n- **International**: Global artist participation\n- **Impact**: Major cultural tourism attraction\n\n### Istanbul Theatre Festival (October)\nDrama and performance arts:\n- **Duration**: Month-long celebration\n- **Venues**: Historic and contemporary theaters\n- **Content**: Local and international productions\n- **Languages**: Turkish and subtitled performances\n- **Innovation**: Experimental and classical works\n\n## ❄️ Winter Celebrations\n\n### New Year Celebrations\nModern Istanbul party atmosphere:\n- **Venues**: Taksim Square, Ortaköy, Beşiktaş\n- **Fireworks**: Bosphorus Bridge displays\n- **Parties**: Rooftop celebrations, club events\n- **Atmosphere**: International, cosmopolitan\n- **Public transport**: Extended service hours\n\n### Religious Festivals\nIslamic calendar celebrations:\n- **Ramazan**: Month-long fasting period\n- **Iftar**: Evening fast-breaking community meals\n- **Eid celebrations**: Major holiday periods\n- **Cultural aspects**: Non-religious participation welcome\n- **Food**: Special holiday dishes and sweets\n\n## 🎨 Year-Round Cultural Events\n\n### Contemporary Istanbul (November)\nArt fair and cultural week:\n- **Focus**: Contemporary art, design, culture\n- **Venues**: Multiple galleries and cultural spaces\n- **International**: Regional and global participants\n- **Commercial**: Art sales and collector events\n- **Education**: Artist talks, panel discussions\n\n### Film Festivals\nCinematic celebrations:\n- **Istanbul International Film Festival**: April\n- **Documentary film festivals**: Various months\n- **Venues**: Historic theaters, modern cinemas\n- **International**: Global cinema representation\n- **Awards**: Recognition for Turkish and international films\n\n## 📅 Planning Your Festival Visit\n\n### Booking and Preparation\n- **Advance tickets**: Popular events sell out quickly\n- **Accommodation**: Higher rates during major festivals\n- **Transportation**: Extra crowds on public transport\n- **Weather**: Dress appropriately for outdoor events\n- **Language**: English information usually available\n\n### Local Festival Culture\n- **Community participation**: Locals actively attend events\n- **Family atmosphere**: Many festivals welcome children\n- **Food**: Special festival foods and vendors\n- **Socializing**: Festivals are social experiences\n- **Photography**: Generally welcome at public events\n\nIstanbul's festival calendar reflects the city's role as a cultural bridge, offering both traditional Turkish celebrations and international contemporary arts events.	Cultural Events Guide	\N	2025-09-23 22:28:00	5
20	Sunset and Sunrise Spots: Golden Hour Photography	# Chasing Golden Light: Best Sunset and Sunrise Locations\n\nIstanbul's unique geography creates spectacular opportunities for golden hour photography, with water reflections, silhouetted minarets, and dramatic skies providing endless creative possibilities.\n\n## 🌅 Prime Sunrise Locations\n\n### Üsküdar Waterfront\nAsian side sunrise magic:\n- **View**: European side silhouettes against morning light\n- **Landmarks**: Maiden's Tower, Topkapı Palace outline\n- **Accessibility**: Easy ferry access from European side\n- **Photography**: Golden light on Bosphorus waters\n- **Timing**: 6:00-7:30 AM depending on season\n- **Peaceful**: Fewer crowds than evening locations\n\n### Çamlıca Hill\nHighest vantage point:\n- **Elevation**: 360-degree city panorama\n- **View**: Entire Istanbul layout visible\n- **Facilities**: Observation deck, cafe\n- **Transportation**: Metro plus taxi/bus\n- **Best season**: Clear weather days\n- **Equipment**: Wide-angle lens recommended\n\n## 🌇 Iconic Sunset Viewpoints\n\n### Galata Tower Vicinity\nClassic Istanbul sunset:\n- **View**: Golden Horn, Old City, minarets\n- **Accessibility**: Easy metro access to area\n- **Options**: Tower itself or nearby rooftop bars\n- **Timing**: 1 hour before sunset for best positioning\n- **Crowds**: Very popular, arrive early\n- **Alternatives**: Nearby streets offer similar views\n\n### Pierre Loti Hill\nRomantic Golden Horn sunset:\n- **Transport**: Cable car or steep walk\n- **View**: 180-degree Golden Horn panorama\n- **Atmosphere**: Historic tea gardens\n- **Facilities**: Traditional tea service\n- **Photography**: Multiple terrace levels\n- **Romance**: Popular proposal location\n\n## 🌊 Waterfront Golden Hours\n\n### Ortaköy Waterfront\nBosphorus Bridge sunset:\n- **Landmark**: Bosphorus Bridge in frame\n- **Foreground**: Historic Ortaköy Mosque\n- **Activities**: Street food, vendors\n- **Accessibility**: Easy public transport\n- **Parking**: Limited, public transport recommended\n- **Social**: Popular with locals and tourists\n\n### Bebek Bay\nSophisticated sunset dining:\n- **Setting**: Crescent-shaped bay\n- **Dining**: Waterfront restaurants with views\n- **Photography**: Reflected light on calm water\n- **Atmosphere**: Upscale, trendy location\n- **Activities**: Jogging path, park areas\n- **Seasonal**: Beautiful in all seasons\n\n## 🏛️ Historic Location Golden Hours\n\n### Sultanahmet Area\nSunrise over historic peninsula:\n- **View**: Blue Mosque, Hagia Sophia silhouettes\n- **Light**: Morning light illuminates historic architecture\n- **Peaceful**: Fewer tourists early morning\n- **Photography**: Architectural details in soft light\n- **Walking**: Easy exploration after sunrise\n- **Breakfast**: Traditional Turkish breakfast nearby\n\n### Süleymaniye Mosque Complex\nSunset over Golden Horn:\n- **Elevation**: Hill location provides overview\n- **Architecture**: Sinan's masterpiece in golden light\n- **View**: Golden Horn, Galata Tower\n- **Atmosphere**: Peaceful, spiritual setting\n- **Respect**: Modest dress, quiet behavior required\n- **Photography**: Stunning architectural details\n\n## 📸 Photography Tips\n\n### Equipment Recommendations\n- **Camera**: DSLR or mirrorless for best quality\n- **Lenses**: Wide-angle for landscapes, telephoto for details\n- **Tripod**: Essential for sharp shots in low light\n- **Filters**: Polarizing and neutral density filters\n- **Backup**: Extra batteries, memory cards\n\n### Technical Settings\n- **Golden hour timing**: 1 hour before/after sunset/sunrise\n- **Exposure**: Bracket shots for HDR processing\n- **Focus**: Manual focus for sharp foregrounds\n- **Composition**: Rule of thirds, leading lines\n- **Weather**: Check forecasts for clear skies\n\n## 🎯 Seasonal Considerations\n\n### Best Months for Photography\n- **Spring**: Clear air, mild temperatures\n- **Autumn**: Dramatic skies, comfortable weather\n- **Winter**: Moody atmospheres, fewer crowds\n- **Summer**: Long days but hazy conditions\n\n### Weather and Light Quality\n- **Clear days**: Sharp, defined shadows\n- **Cloudy skies**: Dramatic, diffused light\n- **After rain**: Clearest air, vibrant colors\n- **Smog**: Can create atmospheric effects\n- **Wind**: Affects water reflections\n\nIstanbul's golden hours offer unparalleled photography opportunities, where ancient architecture meets natural beauty in spectacular displays of light and shadow.	Photography Expert	\N	2025-09-23 22:29:00	9
21	Local Sports Culture: Football, Basketball, and Beyond	# Istanbul Sports Passion: Beyond the Beautiful Game\n\nIstanbul's sports culture runs deep, with passionate fan bases, historic rivalries, and athletic traditions that offer visitors unique insights into local community identity and social dynamics.\n\n## ⚽ Football Culture\n\n### The Big Three\nIstanbul's football giants:\n- **Galatasaray**: Founded 1905, European Cup winners\n- **Fenerbahçe**: Founded 1907, passionate Yellow Canaries\n- **Beşiktaş**: Founded 1903, the Black Eagles\n- **Rivalries**: Intense derbies, city-wide celebrations\n- **European success**: Champions League and UEFA Cup victories\n\n### Stadium Experiences\nMatchday atmosphere:\n- **Türk Telekom Stadium**: Galatasaray's modern home\n- **Şükrü Saracoğlu Stadium**: Fenerbahçe's historic ground\n- **Beşiktaş Park**: Beşiktaş's atmospheric new venue\n- **Fan culture**: Choreographed displays, passionate singing\n- **Derby days**: Entire city energizes around matches\n\n## 🏀 Basketball Excellence\n\n### Professional Teams\nEuropean basketball powerhouses:\n- **Anadolu Efes**: EuroLeague champions\n- **Fenerbahçe Basketball**: Multiple European titles\n- **Galatasaray Basketball**: Historic Turkish basketball\n- **Venues**: Modern arenas with passionate crowds\n- **International**: NBA players in Turkish league\n\n### Basketball Culture\n- **Youth development**: Strong grassroots programs\n- **Court access**: Public basketball courts throughout city\n- **American influence**: NBA popularity growing\n- **Women's basketball**: Competitive professional leagues\n- **Street basketball**: Informal games in neighborhoods\n\n## 🏊 Water Sports\n\n### Bosphorus Activities\nWater sports opportunities:\n- **Swimming**: Bosphorus cross-continental swim\n- **Sailing**: Active sailing clubs and regattas\n- **Rowing**: Historic rowing clubs, competitive teams\n- **Fishing**: Traditional and sport fishing culture\n- **Water polo**: Club and school competitions\n\n### Seasonal Water Sports\n- **Summer**: Peak season for all water activities\n- **Spring/Autumn**: Ideal conditions for sailing\n- **Winter**: Hardy swimmers continue year-round\n- **Clubs**: Historic sports clubs on Bosphorus shores\n- **Events**: International swimming competitions\n\n## 🏃 Running and Fitness Culture\n\n### Popular Running Routes\nScenic fitness opportunities:\n- **Bosphorus coastal path**: Waterfront running with views\n- **Princes Islands**: Car-free island running\n- **Belgrad Forest**: Trail running and hiking\n- **Maçka Park**: Central urban running circuit\n- **Marathon events**: Istanbul Marathon crosses continents\n\n### Fitness Trends\nModern wellness culture:\n- **Gym culture**: Growing fitness center popularity\n- **Yoga**: Outdoor yoga in parks, studio classes\n- **Cycling**: Increasing bicycle infrastructure\n- **Rock climbing**: Indoor climbing gyms\n- **CrossFit**: International fitness trends adopted\n\n## 🎾 Other Sports\n\n### Tennis and Individual Sports\n- **Tennis clubs**: Private and public facilities\n- **Istanbul Open**: ATP tennis tournament\n- **Golf**: Growing golf course development\n- **Martial arts**: Traditional and modern fighting sports\n- **Athletics**: Track and field competitions\n\n### Traditional Turkish Sports\n- **Oil wrestling**: Historic Kirkpinar tradition\n- **Archery**: Traditional Turkish bow techniques\n- **Horseback riding**: Equestrian clubs and facilities\n- **Sailing**: Traditional Turkish sailing boats\n\n## 🏟️ Sports Tourism\n\n### Attending Games\nVisitor game experience:\n- **Ticket purchasing**: Online or at stadium box offices\n- **Atmosphere**: Incredibly passionate fan experiences\n- **Safety**: Generally safe with proper preparation\n- **Transport**: Special transport on match days\n- **Cultural experience**: Understanding local identity\n\n### Sports Bars and Viewing\n- **Sports bars**: Modern venues with multiple screens\n- **Traditional tea houses**: Local men watching matches\n- **Outdoor screening**: Public viewing areas for major games\n- **Community**: Sports as social bonding activity\n\n## 🎯 Sports Calendar Highlights\n\n### Annual Events\n- **Istanbul Marathon**: October/November, intercontinental race\n- **Football derbies**: Season-long intense rivalries\n- **Basketball Final Four**: European basketball excellence\n- **Swimming competitions**: Bosphorus crossing events\n- **Tennis tournaments**: International competitions\n\nIstanbul's sports culture offers visitors authentic community experiences, from the passion of football derbies to the growing fitness and wellness culture embracing active lifestyles.	Sports Culture Expert	\N	2025-09-23 22:30:00	5
22	Accommodation Guide: From Budget to Luxury	# Where to Stay in Istanbul: Complete Accommodation Guide\n\nIstanbul offers accommodation options for every budget and preference, from backpacker hostels to luxury Bosphorus hotels, each providing different experiences of this magnificent city.\n\n## 🏨 Luxury Hotels\n\n### Bosphorus Palace Hotels\nUltimate luxury experiences:\n- **Four Seasons Bosphorus**: Historic Ottoman palace conversion\n- **Çırağan Palace**: Kempinski luxury on Bosphorus shore\n- **The Ritz-Carlton**: Modern luxury with traditional elements\n- **Views**: Unobstructed Bosphorus and city panoramas\n- **Services**: World-class spa, fine dining, concierge\n- **Pricing**: €300-1000+ per night\n\n### Historic Luxury Properties\nCharacter with comfort:\n- **Hotel Les Ottomans**: Exclusive boutique on Bosphorus\n- **Ajia Hotel**: Asian side luxury with European views\n- **Park Hyatt Istanbul**: Contemporary luxury in Nişantaşı\n- **Atmosphere**: Blend of Ottoman heritage and modern luxury\n- **Amenities**: Rooftop dining, spa facilities, butler service\n\n## 🏩 Boutique and Mid-Range Hotels\n\n### Galata District Boutiques\nCharming neighborhood stays:\n- **Georges Hotel Galata**: Historic building conversion\n- **The Haze Karakoy**: Design-focused modern hotel\n- **Vault Karakoy**: Underground cistern-inspired design\n- **Location**: Walking distance to attractions\n- **Character**: Unique architecture and local atmosphere\n- **Pricing**: €100-250 per night\n\n### Sultanahmet Area Hotels\nHeart of historic district:\n- **Hotel Empress Zoe**: Byzantine-era foundations\n- **Sirkeci Mansion**: Ottoman-style boutique hotel\n- **Blue House**: Traditional architecture, modern comfort\n- **Proximity**: Walking distance to major historic sites\n- **Atmosphere**: Immersed in Byzantine and Ottoman history\n- **Services**: Rooftop terraces, traditional Turkish breakfast\n\n## 🏠 Alternative Accommodations\n\n### Airbnb and Apartments\nLocal living experience:\n- **Neighborhoods**: Galata, Karakoy, Cihangir, Kadıköy\n- **Types**: Ottoman houses, modern apartments, loft spaces\n- **Benefits**: Kitchen facilities, local neighborhood feel\n- **Pricing**: €30-150 per night depending on location\n- **Experience**: Live like a local, use neighborhood markets\n- **Considerations**: Language barriers, local regulations\n\n### Pensions and Guesthouses\nFamily-run authentic experiences:\n- **Character**: Personal service, local knowledge\n- **Locations**: Residential neighborhoods throughout city\n- **Atmosphere**: Homestay feeling, cultural exchange\n- **Services**: Home-cooked meals, local recommendations\n- **Pricing**: €25-80 per night\n- **Language**: Often Turkish-only, but welcoming\n\n## 🎒 Budget Accommodation\n\n### Hostels and Backpacker Options\nBudget-friendly social stays:\n- **Sultanahmet areas**: Historic district hostels\n- **Galata hostels**: Modern facilities, young atmosphere\n- **Kadıköy options**: Asian side budget accommodations\n- **Facilities**: Shared kitchens, common areas, laundry\n- **Social**: International travelers, group activities\n- **Pricing**: €10-25 per night for dorm beds\n\n### Budget Hotels\nPrivate rooms, basic amenities:\n- **Family-run hotels**: Simple, clean accommodations\n- **Locations**: Throughout city, varying quality\n- **Services**: Basic breakfast, tourist information\n- **Atmosphere**: No-frills, functional stays\n- **Pricing**: €20-60 per night\n\n## 📍 Neighborhood Guide\n\n### Where to Stay by Interest\n\n**Historic Sightseeing**: Sultanahmet area\n- **Pros**: Walking distance to major sites\n- **Cons**: Tourist crowds, restaurant prices\n- **Best for**: First-time visitors, history enthusiasts\n\n**Nightlife and Modern Culture**: Beyoğlu/Galata\n- **Pros**: Restaurants, bars, cultural venues\n- **Cons**: Busy, some noise at night\n- **Best for**: Young travelers, culture seekers\n\n**Local Authentic Experience**: Kadıköy\n- **Pros**: Local prices, authentic atmosphere\n- **Cons**: Further from European side attractions\n- **Best for**: Cultural immersion, budget travelers\n\n**Luxury and Shopping**: Nişantaşı/Beşiktaş\n- **Pros**: Upscale area, shopping, restaurants\n- **Cons**: Expensive, less historic character\n- **Best for**: Luxury seekers, business travelers\n\n## 💡 Booking Tips\n\n### Best Practices\n- **Advance booking**: Essential during festival periods\n- **Season considerations**: Summer and spring higher prices\n- **Location vs. price**: Balance convenience with budget\n- **Reviews**: Read recent guest experiences\n- **Cancellation**: Flexible policies recommended\n\n### What to Ask About\n- **Wi-Fi**: Essential for most travelers\n- **Air conditioning**: Important during summer months\n- **Breakfast**: Turkish breakfast often excellent value\n- **Transportation**: Proximity to metro, ferry stations\n- **Views**: Bosphorus views significantly affect pricing\n\nIstanbul's accommodation diversity ensures every traveler finds suitable options, from backpacker adventures to luxury escapes, all offering unique perspectives on this incredible city.	Accommodation Expert	\N	2025-09-23 22:31:00	6
23	Language and Communication: Turkish Basics for Travelers	# Speaking Istanbul: Essential Turkish for Travelers\n\nWhile many Istanbulites speak some English, learning basic Turkish phrases enhances your experience and shows respect for local culture. This guide covers essential communication for confident navigation.\n\n## 👋 Essential Greetings\n\n### Daily Greetings\nBasic social interactions:\n- **Merhaba** (mer-ha-BA) - Hello (universal)\n- **Günaydın** (goon-eye-din) - Good morning\n- **İyi akşamlar** (ee-yee AK-sham-lar) - Good evening\n- **İyi geceler** (ee-yee geh-jeh-ler) - Good night\n- **Hoşça kal** (hosh-cha kal) - Goodbye (to person staying)\n- **Güle güle** (goo-leh goo-leh) - Goodbye (to person leaving)\n\n### Polite Expressions\nCourtesy phrases:\n- **Teşekkür ederim** (teh-shek-KOOR eh-deh-rim) - Thank you\n- **Teşekkürler** (teh-shek-koor-LER) - Thanks (informal)\n- **Rica ederim** (ree-JA eh-deh-rim) - You're welcome\n- **Özür dilerim** (oh-ZOOR dee-leh-rim) - I'm sorry/Excuse me\n- **Pardon** (par-DON) - Excuse me (borrowed from French)\n\n## 🗣️ Basic Communication\n\n### Essential Questions\nGetting information:\n- **Nerede?** (neh-reh-DEH) - Where is?\n- **Ne kadar?** (neh ka-DAR) - How much?\n- **Nasıl?** (na-SIL) - How?\n- **Ne zaman?** (neh za-MAN) - When?\n- **Kaç?** (kach) - How many?\n- **İngilizce biliyor musunuz?** - Do you speak English?\n\n### Useful Responses\nCommon answers:\n- **Evet** (eh-VET) - Yes\n- **Hayır** (ha-YIR) - No\n- **Bilmiyorum** (bil-mee-yo-rum) - I don't know\n- **Anlamıyorum** (an-la-muh-yo-rum) - I don't understand\n- **Yardım ederim** (yar-DIM eh-deh-rim) - I can help\n\n## 🍽️ Restaurant and Food\n\n### Ordering Food\nDining phrases:\n- **Menü, lütfen** (meh-NOO loot-fen) - Menu, please\n- **Bu ne?** (bu NEH) - What is this?\n- **İstiyorum** (is-tee-yo-rum) - I want\n- **Hesap, lütfen** (heh-SAP loot-fen) - Bill, please\n- **Su** (SU) - Water\n- **Çay** (chai) - Tea\n- **Kahve** (kah-VEH) - Coffee\n\n### Dietary Restrictions\nSpecial needs:\n- **Vejetaryen** (veh-jeh-tar-yen) - Vegetarian\n- **Et yemiyorum** (et yeh-mee-yo-rum) - I don't eat meat\n- **Alerjim var** (ah-ler-jeem var) - I have allergies\n- **Acı istemiyorum** (ah-JI is-teh-mee-yo-rum) - I don't want spicy\n\n## 🚌 Transportation\n\n### Getting Around\nTransport vocabulary:\n- **Otogar** (oh-toh-GAR) - Bus station\n- **Metro** (meh-TRO) - Metro/subway\n- **Tramvay** (tram-VAI) - Tram\n- **Vapur** (va-PUR) - Ferry\n- **Taksi** (tak-SEE) - Taxi\n- **Nereye gidiyorsunuz?** - Where are you going?\n- **Burada inecek var** - Someone getting off here\n\n### Directions\nNavigation phrases:\n- **Sağ** (SAH) - Right\n- **Sol** (SOL) - Left\n- **Düz** (dooz) - Straight\n- **Yakın** (ya-KIN) - Near/close\n- **Uzak** (u-ZAK) - Far\n- **Kayboldum** (kai-bol-dum) - I'm lost\n\n## 🛍️ Shopping\n\n### Market Interactions\nShopping phrases:\n- **Ne kadar?** (neh ka-DAR) - How much?\n- **Pahalı** (pa-ha-LI) - Expensive\n- **Ucuz** (u-JUZ) - Cheap\n- **İndirim var mı?** (in-dee-rim var muh) - Is there a discount?\n- **Pazarlık** (pa-zar-LIK) - Bargaining\n- **Son fiyat** (son fee-YAT) - Final price\n\n### Payment\nMoney matters:\n- **Nakit** (na-KIT) - Cash\n- **Kredi kartı** (kreh-DEE kar-tuh) - Credit card\n- **Para üstü** (pa-ra oos-TOO) - Change\n- **Fiş** (FISH) - Receipt\n\n## 🆘 Emergency Phrases\n\n### Getting Help\nUrgent communication:\n- **Yardım!** (yar-DIM) - Help!\n- **Polis** (po-LEES) - Police\n- **Hastane** (has-ta-NEH) - Hospital\n- **Doktor** (dok-TOR) - Doctor\n- **Yangın** (yan-GIN) - Fire\n- **Acil durum** (a-JIL du-rum) - Emergency\n\n### Health Issues\nMedical communication:\n- **Hasta** (has-TA) - Sick\n- **Ağrı** (AH-ruh) - Pain\n- **İlaç** (ee-LACH) - Medicine\n- **Eczane** (ej-za-NEH) - Pharmacy\n\n## 📱 Modern Communication\n\n### Technology Terms\n- **Wi-Fi** (vee-fee) - Wi-Fi\n- **İnternet** (in-ter-NET) - Internet\n- **Şifre** (shif-REH) - Password\n- **Telefon** (teh-leh-FON) - Telephone\n- **Şarj** (sharj) - Charging\n\n## 🎯 Cultural Communication Tips\n\n### Body Language and Gestures\n- **Eye contact**: Shows respect and attention\n- **Handshakes**: Common for greetings\n- **Personal space**: Closer than American/Northern European norms\n- **Pointing**: Use open hand instead of index finger\n- **Head movements**: Up-and-down nod means "no" (opposite of English)\n\n### Cultural Sensitivity\n- **Effort appreciated**: Attempts to speak Turkish warmly received\n- **Patient speakers**: Most Turks patient with language learners\n- **English common**: In tourist areas, but Turkish shows respect\n- **Regional dialects**: Istanbul Turkish is standard Turkish\n- **Formality levels**: More formal language shows respect\n\nLearning basic Turkish demonstrates respect for Istanbul culture and opens doors to more authentic local experiences and genuine connections with residents.	Language Learning Expert	\N	2025-09-23 22:32:00	7
\.


--
-- TOC entry 4475 (class 0 OID 16481)
-- Dependencies: 237
-- Data for Name: chat_history; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.chat_history (id, session_id, user_message, ai_response, "timestamp") FROM stdin;
\.


--
-- TOC entry 4476 (class 0 OID 16488)
-- Dependencies: 238
-- Data for Name: chat_sessions; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.chat_sessions (id, session_id, user_id, started_at, last_activity, messages_count, active_navigation_session, has_navigation, context, is_active, ended_at) FROM stdin;
\.


--
-- TOC entry 4477 (class 0 OID 16497)
-- Dependencies: 239
-- Data for Name: conversation_history; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.conversation_history (id, session_id, user_id, user_message, ai_response, route_data, location_data, navigation_active, "timestamp", intent, entities_extracted) FROM stdin;
\.


--
-- TOC entry 4478 (class 0 OID 16506)
-- Dependencies: 240
-- Data for Name: events; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.events (id, name, venue, date, genre, biletix_id) FROM stdin;
1	Beşiktaş Saturday Market	Beşiktaş	\N	market	\N
2	Kadıköy Tuesday Market	Kadıköy	\N	market	\N
3	Friday Live Music - Beyoğlu	Beyoğlu, İstiklal Caddesi	\N	nightlife	\N
4	Sunday Bosphorus Cruise	Eminönü İskelesi	\N	cultural	\N
5	Istanbul Music Festival	Çeşitli mekanlar / Various venues	\N	festival	\N
6	Istanbul Film Festival	Çeşitli sinemalar / Various cinemas	\N	festival	\N
7	Istanbul Biennial	Çeşitli müzeler ve galeriler	\N	exhibition	\N
8	Ramadan Events	Sultanahmet, Eyüp, Fatih	\N	religious	\N
9	New Year Celebrations	Taksim, Ortaköy, Kadıköy	\N	cultural	\N
10	Tulip Festival	Emirgan Korusu, Gülhane Parkı	\N	festival	\N
11	Istanbul Jazz Festival	Harbiye, Zorlu PSM, açık hava mekanları	\N	festival	\N
12	First Friday Art Walk	Karaköy, Galata	\N	cultural	\N
13	Antique Market	Horhor, Çukurcuma	\N	market	\N
\.


--
-- TOC entry 4479 (class 0 OID 16515)
-- Dependencies: 241
-- Data for Name: feedback_events; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.feedback_events (id, user_id, session_id, event_type, item_id, item_type, metadata, "timestamp", processed) FROM stdin;
\.


--
-- TOC entry 4480 (class 0 OID 16527)
-- Dependencies: 242
-- Data for Name: intent_feedback; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.intent_feedback (id, session_id, user_id, original_query, language, predicted_intent, predicted_confidence, classification_method, latency_ms, is_correct, actual_intent, feedback_type, "timestamp", user_action, used_for_training, review_status, context_data) FROM stdin;
\.


--
-- TOC entry 4481 (class 0 OID 16546)
-- Dependencies: 243
-- Data for Name: item_feature_vectors; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.item_feature_vectors (id, item_id, item_type, embedding_vector, embedding_version, total_views, total_clicks, total_saves, avg_rating, conversion_rate, quality_score, updated_at) FROM stdin;
\.


--
-- TOC entry 4482 (class 0 OID 16554)
-- Dependencies: 244
-- Data for Name: location_history; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.location_history (id, user_id, session_id, latitude, longitude, accuracy, altitude, speed, heading, "timestamp", activity_type, is_navigation_active) FROM stdin;
\.


--
-- TOC entry 4483 (class 0 OID 16562)
-- Dependencies: 245
-- Data for Name: museums; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.museums (id, name, location, hours, ticket_price, highlights) FROM stdin;
1	Heritage Spa	Sultanahmet, Caddesi No:55, Sultanahmet/İstanbul	06:00-22:00	\N	hammam: Ottoman-era bathing ritual in the heart of Sultanahmet. Authentic Ottoman experience
2	Underground Coffee	Beyoğlu, Sokak No:53, Beyoğlu/İstanbul	08:00-22:00	\N	hidden_cafe: Secret garden setting in the heart of Beyoğlu. Vintage interior design
3	Artist Collective	Beyoğlu, Sokak No:89, Beyoğlu/İstanbul	10:00-18:00	\N	art_gallery: International exhibitions in the heart of Beyoğlu. Local and international artists
4	Contemporary Gallery	Beyoğlu, Sokak No:1, Beyoğlu/İstanbul	10:00-18:00	\N	art_gallery: International exhibitions in the heart of Beyoğlu. Artist meet-and-greets
5	Traditional Crafts	Galata, Mahallesi No:37, Galata/İstanbul	10:00-18:00	\N	workshop: Learn traditional crafts in the heart of Galata. Expert instruction
6	Rooftop Hideaway	Galata, Caddesi No:18, Galata/İstanbul	08:00-22:00	\N	hidden_cafe: Secret garden setting in the heart of Galata. Features locally roasted coffee
7	Secret Garden Cafe Galata	Galata, Sokak No:2, Galata/İstanbul	08:00-22:00	\N	hidden_cafe: Local favorite coffee spot in the heart of Galata. Vintage interior design
8	Pottery Studio Galata	Galata, Sokak No:47, Galata/İstanbul	10:00-18:00	\N	workshop: Hands-on cultural experience in the heart of Galata. Take home your creations
9	Panoramic Viewpoint Galata	Galata, Caddesi No:68, Galata/İstanbul	24 hours	\N	viewpoint: Breathtaking city views in the heart of Galata. Popular photography spot
10	Coastal Trail	Karaköy, Mahallesi No:98, Karaköy/İstanbul	24 hours	\N	waterfront: Harbor views in the heart of Karaköy. Maritime views
11	Underground Club Karaköy	Karaköy, Caddesi No:66, Karaköy/İstanbul	20:00-04:00	\N	nightlife: Live music performances in the heart of Karaköy. Local crowd
12	Bosphorus Views Karaköy	Karaköy, Sokak No:36, Karaköy/İstanbul	17:00-02:00	\N	rooftop_bar: Stunning city views in the heart of Karaköy. DJ performances on weekends
13	Sky Lounge	Beşiktaş, Sokak No:25, Beşiktaş/İstanbul	17:00-02:00	\N	rooftop_bar: Stunning city views in the heart of Beşiktaş. 360-degree city views
14	Cloud Nine	Beşiktaş, Sokak No:61, Beşiktaş/İstanbul	17:00-02:00	\N	rooftop_bar: Bosphorus panorama in the heart of Beşiktaş. Craft cocktails
15	Neighborhood Market Kadıköy	Kadıköy, Caddesi No:29, Kadıköy/İstanbul	08:00-19:00	\N	market: Traditional market atmosphere in the heart of Kadıköy. Traditional atmosphere
16	Modern Art Space	Kadıköy, Caddesi No:50, Kadıköy/İstanbul	10:00-18:00	\N	art_gallery: Local artist showcase in the heart of Kadıköy. Local and international artists
17	Artisan Bazaar	Kadıköy, Caddesi No:33, Kadıköy/İstanbul	08:00-19:00	\N	market: Authentic shopping experience in the heart of Kadıköy. Traditional atmosphere
18	Underground Coffee	Kadıköy, Sokak No:82, Kadıköy/İstanbul	08:00-22:00	\N	hidden_cafe: Secret garden setting in the heart of Kadıköy. Vintage interior design
19	Creative Hub	Kadıköy, Mahallesi No:89, Kadıköy/İstanbul	10:00-18:00	\N	art_gallery: International exhibitions in the heart of Kadıköy. Rotating monthly exhibitions
20	Panoramic Viewpoint	Üsküdar, Caddesi No:43, Üsküdar/İstanbul	24 hours	\N	viewpoint: Sunset watching spot in the heart of Üsküdar. Popular photography spot
21	Observation Deck Üsküdar	Üsküdar, Sokak No:25, Üsküdar/İstanbul	24 hours	\N	viewpoint: Perfect for photography in the heart of Üsküdar. Best sunset views in the area
22	Heritage Center	Üsküdar, Caddesi No:26, Üsküdar/İstanbul	09:00-17:00	\N	cultural_center: Local cultural activities in the heart of Üsküdar. Community gathering place
23	Heritage Spa	Üsküdar, Sokak No:63, Üsküdar/İstanbul	06:00-22:00	\N	hammam: Ottoman-era bathing ritual in the heart of Üsküdar. Professional masseurs
24	Panoramic Viewpoint Üsküdar	Üsküdar, Sokak No:28, Üsküdar/İstanbul	24 hours	\N	viewpoint: Perfect for photography in the heart of Üsküdar. Popular photography spot
25	Live Music Venue	Şişli, Caddesi No:59, Şişli/İstanbul	20:00-04:00	\N	nightlife: Live music performances in the heart of Şişli. Local crowd
26	Vista Bar Şişli	Şişli, Caddesi No:12, Şişli/İstanbul	17:00-02:00	\N	rooftop_bar: Stunning city views in the heart of Şişli. Craft cocktails
27	Secret Garden Cafe Şişli	Şişli, Caddesi No:60, Şişli/İstanbul	08:00-22:00	\N	hidden_cafe: Local favorite coffee spot in the heart of Şişli. Vintage interior design
28	Jazz Lounge Şişli	Şişli, Sokak No:39, Şişli/İstanbul	20:00-04:00	\N	nightlife: Live music performances in the heart of Şişli. Unique atmosphere
29	Contemporary Gallery	Nişantaşı, Mahallesi No:54, Nişantaşı/İstanbul	10:00-18:00	\N	art_gallery: Local artist showcase in the heart of Nişantaşı. Artist meet-and-greets
30	Cloud Nine	Nişantaşı, Sokak No:53, Nişantaşı/İstanbul	17:00-02:00	\N	rooftop_bar: Stunning city views in the heart of Nişantaşı. DJ performances on weekends
31	Contemporary Gallery	Nişantaşı, Sokak No:24, Nişantaşı/İstanbul	10:00-18:00	\N	art_gallery: International exhibitions in the heart of Nişantaşı. Artist meet-and-greets
32	Artist Collective	Nişantaşı, Caddesi No:26, Nişantaşı/İstanbul	10:00-18:00	\N	art_gallery: Local artist showcase in the heart of Nişantaşı. Local and international artists
33	Creative Hub	Nişantaşı, Sokak No:6, Nişantaşı/İstanbul	10:00-18:00	\N	art_gallery: Local artist showcase in the heart of Nişantaşı. Rotating monthly exhibitions
34	Sunset Terrace	Nişantaşı, Caddesi No:39, Nişantaşı/İstanbul	17:00-02:00	\N	rooftop_bar: Bosphorus panorama in the heart of Nişantaşı. Craft cocktails
35	Craft Workshop	Nişantaşı, Sokak No:80, Nişantaşı/İstanbul	10:00-18:00	\N	workshop: Hands-on cultural experience in the heart of Nişantaşı. Take home your creations
36	Hidden Vista	Ortaköy, Caddesi No:12, Ortaköy/İstanbul	24 hours	\N	viewpoint: Breathtaking city views in the heart of Ortaköy. Best sunset views in the area
37	Artisan Bazaar	Ortaköy, Mahallesi No:16, Ortaköy/İstanbul	08:00-19:00	\N	market: Local produce and crafts in the heart of Ortaköy. Handmade crafts
38	Marble Bath Ortaköy	Ortaköy, Sokak No:45, Ortaköy/İstanbul	06:00-22:00	\N	hammam: Traditional marble hammam in the heart of Ortaköy. Traditional marble construction
39	City Overlook	Ortaköy, Sokak No:31, Ortaköy/İstanbul	24 hours	\N	viewpoint: Sunset watching spot in the heart of Ortaköy. Best sunset views in the area
40	Artisan Bazaar Ortaköy	Ortaköy, Caddesi No:93, Ortaköy/İstanbul	08:00-19:00	\N	market: Local produce and crafts in the heart of Ortaköy. Fresh local produce
41	Heritage Spa	Ortaköy, Caddesi No:46, Ortaköy/İstanbul	06:00-22:00	\N	hammam: Traditional marble hammam in the heart of Ortaköy. Traditional marble construction
42	Arts Center	Balat, Caddesi No:95, Balat/İstanbul	09:00-17:00	\N	cultural_center: Traditional workshops in the heart of Balat. Community gathering place
43	Rooftop Hideaway	Balat, Sokak No:79, Balat/İstanbul	08:00-22:00	\N	hidden_cafe: Local favorite coffee spot in the heart of Balat. Vintage interior design
44	Avant-Garde	Balat, Caddesi No:92, Balat/İstanbul	10:00-18:00	\N	art_gallery: Local artist showcase in the heart of Balat. Rotating monthly exhibitions
45	Traditional Crafts	Balat, Mahallesi No:20, Balat/İstanbul	10:00-18:00	\N	workshop: Learn traditional crafts in the heart of Balat. Expert instruction
46	Textile Workshop	Balat, Mahallesi No:31, Balat/İstanbul	10:00-18:00	\N	workshop: Hands-on cultural experience in the heart of Balat. Expert instruction
47	Pottery Studio	Fener, Caddesi No:58, Fener/İstanbul	10:00-18:00	\N	workshop: Learn traditional crafts in the heart of Fener. Expert instruction
48	Panoramic Viewpoint	Fener, Mahallesi No:63, Fener/İstanbul	24 hours	\N	viewpoint: Perfect for photography in the heart of Fener. Best sunset views in the area
49	Artisan Studio	Fener, Mahallesi No:60, Fener/İstanbul	10:00-18:00	\N	workshop: Master artisan instruction in the heart of Fener. Small group sessions
50	Marble Bath	Eminönü, Sokak No:54, Eminönü/İstanbul	06:00-22:00	\N	hammam: Traditional marble hammam in the heart of Eminönü. Professional masseurs
51	Fresh Market Eminönü	Eminönü, Caddesi No:1, Eminönü/İstanbul	08:00-19:00	\N	market: Local produce and crafts in the heart of Eminönü. Handmade crafts
52	Fresh Market	Eminönü, Mahallesi No:98, Eminönü/İstanbul	08:00-19:00	\N	market: Local produce and crafts in the heart of Eminönü. Traditional atmosphere
53	Waterfront Promenade	Eminönü, Caddesi No:32, Eminönü/İstanbul	24 hours	\N	waterfront: Maritime atmosphere in the heart of Eminönü. Scenic walking paths
54	Artisan Bazaar Eminönü	Eminönü, Mahallesi No:52, Eminönü/İstanbul	08:00-19:00	\N	market: Authentic shopping experience in the heart of Eminönü. Fresh local produce
55	Marina Walk	Eminönü, Caddesi No:18, Eminönü/İstanbul	24 hours	\N	waterfront: Harbor views in the heart of Eminönü. Scenic walking paths
56	Authentic Turkish Bath	Fatih, Caddesi No:51, Fatih/İstanbul	06:00-22:00	\N	hammam: Ottoman-era bathing ritual in the heart of Fatih. Professional masseurs
57	Cultural Foundation	Fatih, Caddesi No:71, Fatih/İstanbul	09:00-17:00	\N	cultural_center: Community events in the heart of Fatih. Community gathering place
58	Heritage Spa Fatih	Fatih, Caddesi No:59, Fatih/İstanbul	06:00-22:00	\N	hammam: Authentic Turkish bath experience in the heart of Fatih. Traditional marble construction
59	Heritage Spa	Fatih, Caddesi No:38, Fatih/İstanbul	06:00-22:00	\N	hammam: Traditional marble hammam in the heart of Fatih. Traditional marble construction
60	Seaside Path	Bakırköy, Sokak No:93, Bakırköy/İstanbul	24 hours	\N	waterfront: Maritime atmosphere in the heart of Bakırköy. Scenic walking paths
\.


--
-- TOC entry 4484 (class 0 OID 16570)
-- Dependencies: 246
-- Data for Name: navigation_events; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.navigation_events (id, session_id, user_id, event_type, event_data, latitude, longitude, current_step, step_instruction, distance_to_next_step, "timestamp") FROM stdin;
\.


--
-- TOC entry 4485 (class 0 OID 16580)
-- Dependencies: 247
-- Data for Name: navigation_sessions; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.navigation_sessions (id, session_id, user_id, chat_session_id, origin_lat, origin_lon, origin_name, destination_lat, destination_lon, destination_name, waypoints, total_distance, total_duration, transport_mode, current_step_index, steps_completed, distance_remaining, time_remaining, status, is_active, route_geometry, route_steps, started_at, completed_at, last_update, actual_duration, deviations_count, reroutes_count) FROM stdin;
\.


--
-- TOC entry 4486 (class 0 OID 16590)
-- Dependencies: 248
-- Data for Name: online_learning_models; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.online_learning_models (id, model_name, model_version, model_type, parameters, hyperparameters, metrics, is_active, is_deployed, deployment_percentage, created_at, updated_at) FROM stdin;
\.


--
-- TOC entry 4487 (class 0 OID 16597)
-- Dependencies: 249
-- Data for Name: places; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.places (id, name, category, district) FROM stdin;
\.


--
-- TOC entry 4488 (class 0 OID 16601)
-- Dependencies: 250
-- Data for Name: restaurants; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.restaurants (id, name, cuisine, location, rating, source, description, place_id, phone, website, price_level, photo_url, photo_reference) FROM stdin;
1	Orient Express & Spa by Orka Hotels	\N	Sultanahmet, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
2	GLK PREMIER Regency Suites & Spa	\N	Sultanahmet, Cankurtaran, Akbıyık Cd. No:46, 34122 Fatih/İstanbul, Türkiye	4.6	local_database	\N	\N	0530 568 25 68	https://www.regencysuitesistanbul.com/	2	\N	\N
3	Gülhanepark Hotel & Spa	\N	Sultanahmet, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
4	Hotel Spectra	\N	Sultanahmet, Binbirdirek, Şht. Mehmetpaşa Ykş. No:2, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 516 35 46	http://www.hotelspectra.com/	2	\N	\N
5	Legacy Ottoman Hotel	\N	Sultanahmet, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
6	GLK PREMIER The Home Suites & Spa	\N	Sultanahmet, Küçük Ayasofya, Küçük Ayasofya Cd. No:60, 34122 Fatih/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 638 88 01	http://www.thehomesuites.com/	2	\N	\N
7	Orka Royal Hotel	\N	Sultanahmet, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
8	Pierre Loti Hotel	\N	Sultanahmet, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
9	Antik Hotel Istanbul	\N	Sultanahmet, Mimar Kemalettin Mahallesi, Beyazıt, Sekbanbaşı Sk. No:6, 34130 Fatih/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 638 58 58	http://www.antikhotel.com/	2	\N	\N
10	Blue House Hotel & Rooftop – 360 Sunset View of Istanbul	\N	Sultanahmet, Cankurtaran, Dalbastı Sk. No:14, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 638 90 10	https://www.bluehouseistanbul.com/	2	\N	\N
11	Viva Deluxe Hotel	\N	Sultanahmet, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
12	Matbah Restaurant	\N	Sultanahmet, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
13	Tria Hotel Istanbul	\N	Sultanahmet, Cankurtaran, Terbıyık Sk. No:7, 34122 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 45 18	http://www.triahotelistanbul.com/	2	\N	\N
14	Prestige Hotel Istanbul Laleli	\N	Sultanahmet, Mimar Kemalettin, Koska Cd. No: 6, 34130 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 82 80	http://www.hotelprestige.com.tr/	2	\N	\N
15	Yasmak Sultan Hotel	\N	Sultanahmet, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
16	Istanbul Town Hotel	\N	Sultanahmet, Mimar Kemalettin, Kalaycı Şevki Sk. No:4, 34126 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 458 00 27	https://www.istanbultownhotel.com/	2	\N	\N
17	Akgun Hotel Beyazit	\N	Sultanahmet, Mimar Kemalettin, Haznedar Sk. No:6, 34490 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 638 28 38	http://www.akgunotel.com.tr/	2	\N	\N
18	Darkhill Hotel	\N	Sultanahmet, Mimar Kemalettin, Koca Ragıppaşa Cd No:9, 34130 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 638 89 00	http://www.darkhillhotel.com/	2	\N	\N
19	Hotel Zurich Istanbul	\N	Sultanahmet, Balabanağa, Vidinli Tevfikpaşa Cd No:14, 34134 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 512 23 50	http://www.hotelzurichistanbul.com/	2	\N	\N
20	GLK PREMIER Acropol Suites	\N	Sultanahmet, Sultanahmet (Old City, Cankurtaran, Akbıyık Cd. No:21, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 638 90 21	https://www.acropolhotel.com/	2	\N	\N
21	Georges Hotel Galata	\N	Beyoğlu, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
22	Nova Plaza Orion Hotel	\N	Beyoğlu, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
23	TAKSİM STAR HOTEL BOSPHORUS	\N	Beyoğlu, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
24	Ramada By Wyndham Istanbul Pera	\N	Beyoğlu, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
25	CVK Park Bosphorus Hotel Istanbul	\N	Beyoğlu, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
26	Avantgarde Urban Hotel Taksim	\N	Beyoğlu, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
27	dora hotel	\N	Beyoğlu, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
28	Ferman Hilal Hotel	\N	Beyoğlu, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
29	Anemon Koleksiyon Galata Otel	\N	Beyoğlu, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
30	360 Istanbul	\N	Beyoğlu, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
31	Orka Taksim Suites & Hotel	\N	Beyoğlu, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
32	Nippon Hotel	\N	Beyoğlu, Kocatepe, Topçu Cd. No:6, 34437 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 313 33 00	http://www.nipponhotel.com.tr/	2	\N	\N
33	BellaVista Hostel	\N	Beyoğlu, Bülbül, Turan Cd. No:50, 34400 Beyoğlu/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 361 92 55	\N	2	\N	\N
34	Opera Hotel Bosphorus	\N	Beyoğlu, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
35	Buyuk Londra Hotel's Terrace Bar	\N	Beyoğlu, Asmalı Mescit, Meşrutiyet Cd. No:53, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 245 06 70	https://londrahotel.net/tr/	2	\N	\N
36	Konak Hotel	\N	Beyoğlu, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
37	Mr CAS Hotels	\N	Beyoğlu, Kuloğlu, İstiklal Cd. No:153, 34433 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 00 07	https://www.mrcashotels.com/	2	\N	\N
38	Van Kahvaltı Evi	\N	Beyoğlu, Kılıçali Paşa, Defterdar Ykş. 52/A, 34425 Beyoğlu/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 293 64 37	\N	2	\N	\N
39	5. Kat Restaurant	\N	Beyoğlu, Cihangir, Soğancı Sk. No:3, 34427, 34433 Beyoğlu/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 293 37 74	http://www.5kat.com/	2	\N	\N
40	Mikla	\N	Beyoğlu, The Marmara Pera, Asmalı Mescit, Meşrutiyet Cd. No:15, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 56 56	http://www.miklarestaurant.com/	2	\N	\N
41	Orient Express & Spa by Orka Hotels	\N	Galata, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
42	Georges Hotel Galata	\N	Galata, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
43	Gülhanepark Hotel & Spa	\N	Galata, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
44	Legacy Ottoman Hotel	\N	Galata, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
45	Orka Royal Hotel	\N	Galata, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
46	TAKSİM STAR HOTEL BOSPHORUS	\N	Galata, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
47	Viva Deluxe Hotel	\N	Galata, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
48	Nova Plaza Orion Hotel	\N	Galata, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
49	Ramada By Wyndham Istanbul Pera	\N	Galata, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
50	Yasmak Sultan Hotel	\N	Galata, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
51	CVK Park Bosphorus Hotel Istanbul	\N	Galata, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
52	Anemon Koleksiyon Galata Otel	\N	Galata, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
53	Matbah Restaurant	\N	Galata, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
54	360 Istanbul	\N	Galata, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
55	Pierre Loti Hotel	\N	Galata, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
56	Avantgarde Urban Hotel Taksim	\N	Galata, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
57	Orka Taksim Suites & Hotel	\N	Galata, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
58	Ferman Hilal Hotel	\N	Galata, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
59	Hotel Vicenza	\N	Galata, Kemal Paşa, Fevziye Cd. No:3, 34134 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 520 17 01	http://www.hotelvicenzaistanbul.com/	2	\N	\N
60	Golden Horn Hotel	\N	Galata, Hoca Paşa, Ebussuud Cd. No:26, 34110 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 74 74	http://www.thegoldenhorn.com/	2	\N	\N
61	Orient Express & Spa by Orka Hotels	\N	Karaköy, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
62	Georges Hotel Galata	\N	Karaköy, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
63	Gülhanepark Hotel & Spa	\N	Karaköy, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
64	Legacy Ottoman Hotel	\N	Karaköy, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
65	Orka Royal Hotel	\N	Karaköy, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
66	TAKSİM STAR HOTEL BOSPHORUS	\N	Karaköy, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
67	Nova Plaza Orion Hotel	\N	Karaköy, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
68	Viva Deluxe Hotel	\N	Karaköy, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
69	Ramada By Wyndham Istanbul Pera	\N	Karaköy, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
70	CVK Park Bosphorus Hotel Istanbul	\N	Karaköy, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
71	Yasmak Sultan Hotel	\N	Karaköy, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
72	Anemon Koleksiyon Galata Otel	\N	Karaköy, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
73	Matbah Restaurant	\N	Karaköy, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
74	Avantgarde Urban Hotel Taksim	\N	Karaköy, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
75	360 Istanbul	\N	Karaköy, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
76	Pierre Loti Hotel	\N	Karaköy, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
77	Orka Taksim Suites & Hotel	\N	Karaköy, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
78	Ferman Hilal Hotel	\N	Karaköy, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
79	Golden Horn Hotel	\N	Karaköy, Hoca Paşa, Ebussuud Cd. No:26, 34110 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 74 74	http://www.thegoldenhorn.com/	2	\N	\N
80	Buyuk Londra Hotel's Terrace Bar	\N	Karaköy, Asmalı Mescit, Meşrutiyet Cd. No:53, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 245 06 70	https://londrahotel.net/tr/	2	\N	\N
81	Swissôtel The Bosphorus Istanbul	\N	Beşiktaş, Vişnezade, Acısu Sk. NO 19, 34357 Beşiktaş/İstanbul, Türkiye	4.8	local_database	\N	\N	(0212) 326 11 00	https://www.swissotel.com/hotels/istanbul/?goto=fiche_hotel&code_hotel=A5D2&merchantid=seo-maps-TR-A5D2&sourceid=aw-cen&utm_medium=seo%20maps&utm_source=google%20Maps&utm_campaign=seo%20maps	2	\N	\N
82	Symbola Bosphorus Istanbul	\N	Beşiktaş, Ortaköy, Portakal Ykş. Cd. No:17, 34347 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	0533 136 37 67	http://www.symbolabosphorus.com/	2	\N	\N
83	CVK Park Bosphorus Hotel Istanbul	\N	Beşiktaş, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
84	Opera Hotel Bosphorus	\N	Beşiktaş, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
85	Feriye Lokantasi	\N	Beşiktaş, Yıldız, Çırağan Cd. No:44, 34347 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 227 22 17	http://www.feriye.com/	2	\N	\N
86	The House Cafe	\N	Beşiktaş, Yıldız, Ortaköy Salhanesi Sk. No:1, 34349 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 227 26 99	\N	2	\N	\N
87	Mado Ortaköy	\N	Beşiktaş, Mecidiye Mahallesi, Ortaköy, İskele Sk. No:24, 34347 Beşiktaş/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 259 20 00	https://www.mado.com.tr/	2	\N	\N
88	Konak Hotel	\N	Beşiktaş, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
89	İBB Fethipaşa Sosyal Tesisleri	\N	Beşiktaş, Kuzguncuk Mahallesi Paşa Limanı Caddesi Nacak Sokak No:6 Kapı No: 14/2, Kuzguncuk, 34674 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	444 1 034	https://tesislerimiz.ibb.istanbul/fethipasa-sosyal-tesisi/	2	\N	\N
90	Topaz İstanbul	\N	Beşiktaş, Ömer Avni, İnönü Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	4.2	local_database	\N	\N	0531 329 41 11	http://www.topazistanbul.com/	2	\N	\N
91	ORTAKÖY POLİSEVİ SOSYAL TESİSİ (‼️KONAKLAMA YOKTUR‼️)	\N	Beşiktaş, Ortaköy, Portakal Ykş. Cd. No:58, 34347 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	0505 318 34 76	\N	2	\N	\N
92	Nişantaşı Başköşe	\N	Beşiktaş, Harbiye, Bronz Sk. No:5/1, 34370 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 230 38 68	http://www.nisantasibaskose.com/	2	\N	\N
93	The Brasserie Restaurant	\N	Beşiktaş, 1, Gümüşsuyu, Asker Ocağı Cd., 34437 Beyoğlu/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 368 44 44	http://istanbul.intercontinental.com/	2	\N	\N
94	Ali Baba Iskender Kebapcisi	\N	Beşiktaş, Sinanpaşa, Yeni Hamam Sk. No:11, 34353 Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 227 77 28	http://www.alibabaiskender.com/	2	\N	\N
95	Elma Cafe & Pub	\N	Beşiktaş, Sinanpaşa Köyiçi Cad, Sinanpaşa, Leşker Sk. No:1, 34353 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 261 36 34	http://www.elmacafepub.com/	2	\N	\N
96	Murat Muhallebi Beşiktaş Şube	\N	Beşiktaş, Sinanpaşa, Ihlamurdere Cd. No:10, 34353 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 258 45 20	\N	2	\N	\N
97	Renkli Limon	\N	Beşiktaş, Sinanpaşa, Ihlamurdere Caddesi. Alaybeyi Sok. No:20, 34353 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 258 80 85	\N	2	\N	\N
98	Dragon Restaurant	\N	Beşiktaş, Harbiye, Cumhuriyet Cd. No:50, 34367 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 231 62 00	http://dragonrestaurant.com.tr/	2	\N	\N
99	Okka	\N	Beşiktaş, Akaretler, Vişnezade, Süleyman Seba Cd. No: 22, 34357 Beşiktaş/İstanbul, Türkiye	5	local_database	\N	\N	(0212) 381 21 21	\N	2	\N	\N
100	Sharap	\N	Beşiktaş, Vişnezade, Süleyman Seba Cd. Bjk Plaza A Blok 1 No:48, 34357 Beşiktaş/İstanbul, Türkiye	5	local_database	\N	\N	(0212) 236 93 55	http://193.255.140.26/	2	\N	\N
101	Symbola Bosphorus Istanbul	\N	Ortaköy, Ortaköy, Portakal Ykş. Cd. No:17, 34347 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	0533 136 37 67	http://www.symbolabosphorus.com/	2	\N	\N
102	Avantgarde Urban Hotel Levent	\N	Ortaköy, Esentepe, Büyükdere Cd. No:161, 34394 Şişli/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 337 04 44	https://www.avantgardecollection.com/urban-levent-hotel	2	\N	\N
103	Feriye Lokantasi	\N	Ortaköy, Yıldız, Çırağan Cd. No:44, 34347 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 227 22 17	http://www.feriye.com/	2	\N	\N
104	Sunset Grill & Bar	\N	Ortaköy, Kuruçeşme Mahallesi Ulus Park, Kuruçeşme, Yol Sokağı No:2, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 287 03 57	https://www.sunsetgrillbar.com/	2	\N	\N
105	The House Cafe	\N	Ortaköy, Yıldız, Ortaköy Salhanesi Sk. No:1, 34349 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 227 26 99	\N	2	\N	\N
106	ORTAKÖY POLİSEVİ SOSYAL TESİSİ (‼️KONAKLAMA YOKTUR‼️)	\N	Ortaköy, Ortaköy, Portakal Ykş. Cd. No:58, 34347 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	0505 318 34 76	\N	2	\N	\N
107	Mado Ortaköy	\N	Ortaköy, Mecidiye Mahallesi, Ortaköy, İskele Sk. No:24, 34347 Beşiktaş/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 259 20 00	https://www.mado.com.tr/	2	\N	\N
108	HuQQa Kuruçeşme	\N	Ortaköy, Kuruçeşme, Muallim Naci Cd. No:56, 34345 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 265 06 66	https://www.huqqa.com/tr/kurucesme-huqqa/	2	\N	\N
109	Ulus Cafe	\N	Ortaköy, Kuruçeşme Adnan Saygun Cad, Kuruçeşme, Ulus Parkı 71/B, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 263 84 37	http://uluscafe.com/	2	\N	\N
110	Galatasaray İsland	\N	Ortaköy, Kuruçeşme, Galatasaray Adası, 34000 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	\N	\N	2	\N	\N
111	MEŞHUR BURSA İNEGÖL KÖFTECiSi	\N	Ortaköy, Mecidiye, Dereboyu Cd. No:49A, 34347 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 258 13 12	https://www.tripadvisor.com.tr/Restaurants-g2540453-Inegol.html	2	\N	\N
112	Meshur Adiyaman Cig Koftecisi	\N	Ortaköy, Ortaköy, Dereboyu Cd. No:102 D:b, 34347 Beşiktaş/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 259 96 94	http://www.adiyamancigkoftecisi.com/	2	\N	\N
113	Chinese & Sushi Express	\N	Ortaköy, Ortaköy Ambarlıdere Cad Lotus World No: 6, Ortaköy, D:1, 34400 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	444 1 246	https://www.sushiexpress.com.tr/tr/subeler/lotus-ortakoy	2	\N	\N
114	inari Omakase Kuruçeşme	\N	Ortaköy, Kuruçeşme, Kuruçeşme Cd. No:11, 34345 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 265 96 01	https://www.inariomakase.com/kurucesme/	2	\N	\N
115	Ay Balik	\N	Ortaköy, Kuruçeşme, Kuruçeşme Cd. No:15, 34345 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	0532 417 75 71	\N	2	\N	\N
116	Golden Kokoreç	\N	Ortaköy, Mecidiye, Dereboyu Cd. No:3, 34347 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 259 79 89	\N	2	\N	\N
117	Ali Usta Burma Kadayif & Baklava	\N	Ortaköy, Mecidiye, Dereboyu Cd. No:135A, 34347 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 227 41 90	http://www.aliustabaklava.com/	2	\N	\N
118	Baklavaci Dedeoglu	\N	Ortaköy, Ortaköy, Dereboyu Cd. 61/A, 34347 Beşiktaş/İstanbul, Türkiye	4.7	local_database	\N	\N	(0212) 259 63 10	\N	2	\N	\N
119	Bizce Ortaköy Kuru Fasulyecisi	\N	Ortaköy, Ortaköy, Dereboyu Cd. No:52, 34347 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 236 06 21	https://ortakoykurufasulyecisi.com/	2	\N	\N
120	Dedeoğlu Baklava -Balmumcu	\N	Ortaköy, Balmumcu Mahallesi Mustafa İzzet Efendi sokak no :8, Mecidiye, 34349 Beşiktaş/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 288 78 78	http://www.dedeoglu.com.tr/	2	\N	\N
121	Sarnic Pub & Hotel	\N	Kadıköy, Caferağa, Dumlupınar Sk. No:12, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	0533 729 09 30	http://www.sarnicmeyhaneveotel.com/	2	\N	\N
122	Moda Sea Club 1935	\N	Kadıköy, Moda Caddesi, Caferağa, Ferit Tek Sok. No:1, 34710 Kadıköy/İstanbul, Türkiye	4.5	local_database	\N	\N	(0216) 346 90 72	http://www.modadenizkulubu.org.tr/	2	\N	\N
123	Hacioglu	\N	Kadıköy, Caferağa, Yasa Cd. No:40/A, 34710 Kadıköy/İstanbul, Türkiye	3	local_database	\N	\N	(0216) 414 44 14	https://www.hacioglu.com.tr/	2	\N	\N
124	Görgülü Pastane ve Restaurant	\N	Kadıköy, Acıbadem, Acıbadem Cd. No:94, 34718 Kadıköy/İstanbul, Türkiye	4.1	local_database	\N	\N	(0216) 545 09 02	http://www.gorgulupasta.com/	2	\N	\N
125	Harput Koz Kebap Salonu	\N	Kadıköy, Hasanpaşa, Ulusuluk Sk. No:2, 34722 Kadıköy/İstanbul, Türkiye	3.4	local_database	\N	\N	(0216) 326 71 53	\N	2	\N	\N
126	Mercan Kokoreç	\N	Kadıköy, Caferağa, Muvakkıthane Cd. 15/A, 34710 Kadıköy/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 450 56 56	http://www.mercankokorec.com.tr/	2	\N	\N
127	Pide Sun	\N	Kadıköy, Şükran Apt, Caferağa, Moda Cd. 67/B, 34710 Kadıköy/İstanbul, Türkiye	4.2	local_database	\N	\N	(0216) 450 43 43	http://www.pidesun.com/servis.htm	2	\N	\N
128	Pilav Station	\N	Kadıköy, Caferağa, Piri Çavuş Sk. No:37 D:B, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	(0216) 348 52 56	\N	2	\N	\N
129	Hosta	\N	Kadıköy, Osmanağa, Söğütlü Çeşme Cd No:6, 34714 Kadıköy/İstanbul, Türkiye	2.8	local_database	\N	\N	(0216) 405 16 78	http://www.hosta.com.tr/	2	\N	\N
130	Doyuran Kokoreç	\N	Kadıköy, Rasimpaşa, Rıhtım Cd. No:24, 34716 Kadıköy/İstanbul, Türkiye	3.7	local_database	\N	\N	(0216) 330 47 00	\N	2	\N	\N
131	Karafırın Hasanpaşa	\N	Kadıköy, 13, Hasanpaşa, Fahrettin Kerim Gökay Cd, 34722 Kadıköy/İstanbul, Türkiye	3.4	local_database	\N	\N	(0216) 414 96 50	https://www.karafirin.com.tr/	2	\N	\N
132	Kadı Lokantası	\N	Kadıköy, Osmanağa, Halitağa Cd. No:44, 34714 Kadıköy/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 449 27 17	\N	2	\N	\N
133	Fil Shot	\N	Kadıköy, Caferağa, Sakız Gülü Sk. No:26/A, 34710 Kadıköy/İstanbul, Türkiye	4.1	local_database	\N	\N	0532 651 95 80	http://filbistro.com/	2	\N	\N
134	Hacı Ali 2 Et Lokantası	\N	Kadıköy, Osmanağa, Karadut Sk. No:5, 34714 Kadıköy/İstanbul, Türkiye	4.4	local_database	\N	\N	(0216) 346 32 81	\N	2	\N	\N
135	Kimyon Kadıköy	\N	Kadıköy, Caferağa, Kadife Sk. No:17 D:C, 34710 Kadıköy/İstanbul, Türkiye	4.3	local_database	\N	\N	(0216) 330 48 45	http://www.kimyon.com.tr/	2	\N	\N
136	Burger King - Acıbadem	\N	Kadıköy, Acıbadem, Dar Sk. No:1/1, 34718 Kadıköy/İstanbul, Türkiye	3.4	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
137	Tek Buffet	\N	Kadıköy, Caferağa, Moda Cd. No:70 D:70, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	(0216) 338 17 70	\N	2	\N	\N
138	Pilav Dunyasi	\N	Kadıköy, Caferağa, Sakız Gülü Sk. 21/2, 34710 Kadıköy/İstanbul, Türkiye	2.8	local_database	\N	\N	(0216) 348 50 87	\N	2	\N	\N
139	Çiya Sofrası	\N	Kadıköy, Caferağa, Güneşli Bahçe Sok, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	(0216) 418 51 15	http://www.ciya.com.tr/	2	\N	\N
140	Cafe & Shop Kadıköy	\N	Kadıköy, Caferağa Mh Bahariye Cad, Caferağa, Hacı Şükrü Sk. No:11, 34710 Kadıköy/İstanbul, Türkiye	4.4	local_database	\N	\N	(0216) 418 60 64	\N	2	\N	\N
141	İBB Fethipaşa Sosyal Tesisleri	\N	Üsküdar, Kuzguncuk Mahallesi Paşa Limanı Caddesi Nacak Sokak No:6 Kapı No: 14/2, Kuzguncuk, 34674 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	444 1 034	https://tesislerimiz.ibb.istanbul/fethipasa-sosyal-tesisi/	2	\N	\N
142	Ozsut	\N	Üsküdar, CAPİTOL AVM Altunizade Mahallesi PROF.DR.MAHİR İZ CAD. NO:7 ZEMİN KAT, Altunizade, 34662 Üsküdar/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 651 41 94	http://www.ozsut.com.tr/	2	\N	\N
143	Cafe 5. Cadde	\N	Üsküdar, Salacak, Üsküdar Harem Sahil Yolu No:29, 34000 Üsküdar/İstanbul, Türkiye	3.9	local_database	\N	\N	\N	\N	2	\N	\N
144	Üsküdar Pide	\N	Üsküdar, Valide-i Atik, Çavuşdere Cd. No:86, 34672 Üsküdar/İstanbul, Türkiye	3.8	local_database	\N	\N	(0216) 492 72 72	http://www.meshuristanbulpide.com/	2	\N	\N
145	ÜSKÜDAR DÖNER LAHMACUN PİDE SALONU	\N	Üsküdar, Valide-i Atik, Çavuşdere Cd. 72/A, 34664 Üsküdar/İstanbul, Türkiye	4.2	local_database	\N	\N	(0216) 341 24 98	\N	2	\N	\N
146	Temel Reis Akcaabat Koftecisi	\N	Üsküdar, Kerem Yilmazer Sahnesi, Valide-i Atik, Dr. Fahri Atabey Cd. No:79 Alti, 34668 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	(0216) 553 42 20	http://kuymakadam.com/	2	\N	\N
147	Şişmanoğlu Tantuni	\N	Üsküdar, Ahmediye, Dr. Fahri Atabey Cd., 34672 Üsküdar/İstanbul, Türkiye	4.2	local_database	\N	\N	(0216) 343 23 20	http://sismanoglutantuni.com/	2	\N	\N
148	As Kebap	\N	Üsküdar, Sultantepe, 34664 Üsküdar/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 334 21 24	\N	2	\N	\N
149	Limon Pide Kebap Evi	\N	Üsküdar, Hacı Hesna Hatun Mh, Selmanağa Bostanı Sk. No:8, 34777 Üsküdar/İstanbul, Türkiye	3.6	local_database	\N	\N	(0216) 553 43 43	https://www.limonpidekebapevi.com/	2	\N	\N
150	Altın Şiş Restaurant	\N	Üsküdar, Ahmediye, Kefçedede Mektebi Sokağı No:10, 34672 Üsküdar/İstanbul, Türkiye	4.3	local_database	\N	\N	(0216) 553 23 70	http://altınşiş.com.tr/	2	\N	\N
151	Genc Kebap	\N	Üsküdar, Mimar Sinan, Selmani Pak Cd. No:1, 34674 Üsküdar/İstanbul, Türkiye	3.3	local_database	\N	\N	(0216) 391 96 63	\N	2	\N	\N
152	Ozbolu Lokantasi	\N	Üsküdar, Mimar Sinan, Atlas Sk. No:11, 34672 Üsküdar/İstanbul, Türkiye	4.1	local_database	\N	\N	(0216) 532 73 31	http://ozbolu-lokantasi.com/	2	\N	\N
153	Güneş Kokoreç	\N	Üsküdar, Mimar Sinan, Balaban Cd. No:33, 34672 Üsküdar/İstanbul, Türkiye	4.2	local_database	\N	\N	0541 842 19 37	https://www.instagram.com/guneskokorec1974?igsh=eDRnYWxwajBzMTg4	2	\N	\N
154	Pilav House	\N	Üsküdar, Mimar Sinan, Selami Ali Efendi Cad. 31/A, 34672 Üsküdar/Istanbul - Asia, Türkiye	3.9	local_database	\N	\N	(0216) 391 29 54	\N	2	\N	\N
155	Çiğ Köfteci Ömer Usta	\N	Üsküdar, Mimar Sinan, Selmani Pak Cd. No:32 D:E, 34664 Üsküdar/İstanbul, Türkiye	3.7	local_database	\N	\N	\N	https://cigkofteciomerusta.com/	2	\N	\N
156	Üsküdarlı Beyzade	\N	Üsküdar, Ahmediye, Dr. Fahri Atabey Cd. No:24, 34672 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	\N	\N	2	\N	\N
157	Uluç Ekmek Arası	\N	Üsküdar, Aziz Mahmut Hüdayi, Hakimiyeti Milliye Cd. 58 A, 34672 Üsküdar/İstanbul, Türkiye	4.5	local_database	\N	\N	(0216) 310 77 88	\N	2	\N	\N
158	Burger King - Üsküdar Meydan	\N	Üsküdar, Aziz Mahmut Hüdayi, Hakimiyeti Milliye Cd. No:78, 34672 Üsküdar/İstanbul, Türkiye	3.2	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
159	Hacıoğlu	\N	Üsküdar, Aziz Mahmut Hüdayi, Hakimiyeti Milliye Cd. No:46, 34672 Üsküdar/İstanbul, Türkiye	2.8	local_database	\N	\N	(0216) 391 31 01	https://www.hacioglu.com.tr/	2	\N	\N
160	Beyaz Saray	\N	Üsküdar, Mimar Sinan, Hakimiyeti Milliye Cd. No:22, 34672 Üsküdar/İstanbul, Türkiye	3.1	local_database	\N	\N	(0216) 341 25 08	\N	2	\N	\N
161	Orient Express & Spa by Orka Hotels	\N	Eminönü, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
162	Georges Hotel Galata	\N	Eminönü, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
163	Gülhanepark Hotel & Spa	\N	Eminönü, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
164	GLK PREMIER Regency Suites & Spa	\N	Eminönü, Cankurtaran, Akbıyık Cd. No:46, 34122 Fatih/İstanbul, Türkiye	4.6	local_database	\N	\N	0530 568 25 68	https://www.regencysuitesistanbul.com/	2	\N	\N
165	Hotel Spectra	\N	Eminönü, Binbirdirek, Şht. Mehmetpaşa Ykş. No:2, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 516 35 46	http://www.hotelspectra.com/	2	\N	\N
166	Legacy Ottoman Hotel	\N	Eminönü, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
167	Orka Royal Hotel	\N	Eminönü, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
168	Antik Hotel Istanbul	\N	Eminönü, Mimar Kemalettin Mahallesi, Beyazıt, Sekbanbaşı Sk. No:6, 34130 Fatih/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 638 58 58	http://www.antikhotel.com/	2	\N	\N
169	Pierre Loti Hotel	\N	Eminönü, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
170	GLK PREMIER The Home Suites & Spa	\N	Eminönü, Küçük Ayasofya, Küçük Ayasofya Cd. No:60, 34122 Fatih/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 638 88 01	http://www.thehomesuites.com/	2	\N	\N
171	Prestige Hotel Istanbul Laleli	\N	Eminönü, Mimar Kemalettin, Koska Cd. No: 6, 34130 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 82 80	http://www.hotelprestige.com.tr/	2	\N	\N
172	Blue House Hotel & Rooftop – 360 Sunset View of Istanbul	\N	Eminönü, Cankurtaran, Dalbastı Sk. No:14, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 638 90 10	https://www.bluehouseistanbul.com/	2	\N	\N
173	Viva Deluxe Hotel	\N	Eminönü, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
174	Hotel Vicenza	\N	Eminönü, Kemal Paşa, Fevziye Cd. No:3, 34134 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 520 17 01	http://www.hotelvicenzaistanbul.com/	2	\N	\N
175	Matbah Restaurant	\N	Eminönü, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
176	Hotel Zurich Istanbul	\N	Eminönü, Balabanağa, Vidinli Tevfikpaşa Cd No:14, 34134 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 512 23 50	http://www.hotelzurichistanbul.com/	2	\N	\N
177	Tria Hotel Istanbul	\N	Eminönü, Cankurtaran, Terbıyık Sk. No:7, 34122 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 45 18	http://www.triahotelistanbul.com/	2	\N	\N
178	Ramada By Wyndham Istanbul Pera	\N	Eminönü, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
179	Darkhill Hotel	\N	Eminönü, Mimar Kemalettin, Koca Ragıppaşa Cd No:9, 34130 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 638 89 00	http://www.darkhillhotel.com/	2	\N	\N
180	Yasmak Sultan Hotel	\N	Eminönü, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
181	Orient Express & Spa by Orka Hotels	\N	Fatih, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
182	Georges Hotel Galata	\N	Fatih, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
183	Gülhanepark Hotel & Spa	\N	Fatih, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
184	Legacy Ottoman Hotel	\N	Fatih, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
185	Orka Royal Hotel	\N	Fatih, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
186	Antik Hotel Istanbul	\N	Fatih, Mimar Kemalettin Mahallesi, Beyazıt, Sekbanbaşı Sk. No:6, 34130 Fatih/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 638 58 58	http://www.antikhotel.com/	2	\N	\N
187	Pierre Loti Hotel	\N	Fatih, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
188	Viva Deluxe Hotel	\N	Fatih, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
189	Hotel Vicenza	\N	Fatih, Kemal Paşa, Fevziye Cd. No:3, 34134 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 520 17 01	http://www.hotelvicenzaistanbul.com/	2	\N	\N
190	Prestige Hotel Istanbul Laleli	\N	Fatih, Mimar Kemalettin, Koska Cd. No: 6, 34130 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 82 80	http://www.hotelprestige.com.tr/	2	\N	\N
191	Ramada By Wyndham Istanbul Pera	\N	Fatih, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
192	Hotel Zurich Istanbul	\N	Fatih, Balabanağa, Vidinli Tevfikpaşa Cd No:14, 34134 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 512 23 50	http://www.hotelzurichistanbul.com/	2	\N	\N
193	Hotel Spectra	\N	Fatih, Binbirdirek, Şht. Mehmetpaşa Ykş. No:2, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 516 35 46	http://www.hotelspectra.com/	2	\N	\N
194	Matbah Restaurant	\N	Fatih, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
195	Yasmak Sultan Hotel	\N	Fatih, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
196	Istanbul Town Hotel	\N	Fatih, Mimar Kemalettin, Kalaycı Şevki Sk. No:4, 34126 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 458 00 27	https://www.istanbultownhotel.com/	2	\N	\N
197	Darkhill Hotel	\N	Fatih, Mimar Kemalettin, Koca Ragıppaşa Cd No:9, 34130 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 638 89 00	http://www.darkhillhotel.com/	2	\N	\N
198	Akgun Hotel Beyazit	\N	Fatih, Mimar Kemalettin, Haznedar Sk. No:6, 34490 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 638 28 38	http://www.akgunotel.com.tr/	2	\N	\N
199	Anemon Koleksiyon Galata Otel	\N	Fatih, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
200	Hotel Budo	\N	Fatih, Kemal Paşa, Laleli Cd. No:53, 34470 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 526 27 27	http://www.budohotel.com/	2	\N	\N
201	Georges Hotel Galata	\N	Taksim, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
202	Nova Plaza Orion Hotel	\N	Taksim, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
203	Swissôtel The Bosphorus Istanbul	\N	Taksim, Vişnezade, Acısu Sk. NO 19, 34357 Beşiktaş/İstanbul, Türkiye	4.8	local_database	\N	\N	(0212) 326 11 00	https://www.swissotel.com/hotels/istanbul/?goto=fiche_hotel&code_hotel=A5D2&merchantid=seo-maps-TR-A5D2&sourceid=aw-cen&utm_medium=seo%20maps&utm_source=google%20Maps&utm_campaign=seo%20maps	2	\N	\N
204	TAKSİM STAR HOTEL BOSPHORUS	\N	Taksim, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
205	Ramada By Wyndham Istanbul Pera	\N	Taksim, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
206	CVK Park Bosphorus Hotel Istanbul	\N	Taksim, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
207	Avantgarde Urban Hotel Taksim	\N	Taksim, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
208	dora hotel	\N	Taksim, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
209	Ferman Hilal Hotel	\N	Taksim, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
210	Anemon Koleksiyon Galata Otel	\N	Taksim, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
211	360 Istanbul	\N	Taksim, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
212	Orka Taksim Suites & Hotel	\N	Taksim, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
213	Nippon Hotel	\N	Taksim, Kocatepe, Topçu Cd. No:6, 34437 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 313 33 00	http://www.nipponhotel.com.tr/	2	\N	\N
214	Opera Hotel Bosphorus	\N	Taksim, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
215	BellaVista Hostel	\N	Taksim, Bülbül, Turan Cd. No:50, 34400 Beyoğlu/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 361 92 55	\N	2	\N	\N
216	Konak Hotel	\N	Taksim, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
217	Buyuk Londra Hotel's Terrace Bar	\N	Taksim, Asmalı Mescit, Meşrutiyet Cd. No:53, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 245 06 70	https://londrahotel.net/tr/	2	\N	\N
218	Mr CAS Hotels	\N	Taksim, Kuloğlu, İstiklal Cd. No:153, 34433 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 00 07	https://www.mrcashotels.com/	2	\N	\N
219	Van Kahvaltı Evi	\N	Taksim, Kılıçali Paşa, Defterdar Ykş. 52/A, 34425 Beyoğlu/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 293 64 37	\N	2	\N	\N
220	5. Kat Restaurant	\N	Taksim, Cihangir, Soğancı Sk. No:3, 34427, 34433 Beyoğlu/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 293 37 74	http://www.5kat.com/	2	\N	\N
221	The Cettia Istanbul	\N	Şişli, Fulya Mahallesi, Mecidiyeköy, Ortaklar Cd. No:30, 34834 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 912 43 62	https://www.thecettiaistanbul.com/	2	\N	\N
222	Holiday Inn Istanbul Sisli	\N	Şişli, 19 Mayıs, Halaskargazi Cd. No:206, 34360 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 373 38 00	http://www.hisisli.com/	2	\N	\N
223	dora hotel	\N	Şişli, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
224	Burger King	\N	Şişli, Fulya, Prof. Dr. Bülent Tarcan Sk. No:86, 34394 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
225	Konak Hotel	\N	Şişli, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
226	Domino's Pizza	\N	Şişli, Gayrettepe, Ortaklar Cd. 42/A, 34394 Şişli/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 347 33 99	https://www.dominos.com.tr/subeler/istanbul/gayrettepe-42804	2	\N	\N
227	Domino's Pizza Şişli	\N	Şişli, Akın, Cumhuriyet, Silahşör Cd. No:39 Apt. 37, 34380 Şişli/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 231 48 00	https://www.dominos.com.tr/subeler/istanbul/sisli-42821	2	\N	\N
228	Nişantaşı Başköşe	\N	Şişli, Harbiye, Bronz Sk. No:5/1, 34370 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 230 38 68	http://www.nisantasibaskose.com/	2	\N	\N
229	Burger King	\N	Şişli, 19 Mayıs, Büyükdere Cd. No:24, 34360 Şişli/İstanbul, Türkiye	3.3	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
230	Little Caesars Fulya Şubesi	\N	Şişli, 19 Mayıs Caddesi, 19 Mayıs, Aşçılar Sk. No:3/B, 34394 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 274 35 55	https://www.littlecaesars.com.tr/?utm_source=GMB&utm_medium=local&utm_campaign=LC-Ortaklar-Profile-Click	2	\N	\N
231	Cafe Zone	\N	Şişli, Halaskargazi, Kuyumcu İrfan Sk. 3/B, 34371 Şişli/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 296 65 91	https://www.cafezone.com.tr/	2	\N	\N
232	Garden Iskender	\N	Şişli, Meşrutiyet Mh. Halaskargazi Cd. Ebe Kızı Sk., No:2 Osmanbey, Meşrutiyet, 34363 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 247 45 77	http://gardeniskender.com/	2	\N	\N
233	Dragon Restaurant	\N	Şişli, Harbiye, Cumhuriyet Cd. No:50, 34367 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 231 62 00	http://dragonrestaurant.com.tr/	2	\N	\N
234	Zorba	\N	Şişli, Gayrettepe, Yıldız Posta Cd. No:28 D:g, 34349 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 267 13 12	http://www.zorbataverna.com.tr/	2	\N	\N
235	Rumeli Köftecisi	\N	Şişli, Fulya, Bahçeler Sokağı No:9, 34394 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 274 38 84	\N	2	\N	\N
236	Özlem Kebap	\N	Şişli, Kurtuluş Mh, Kurtuluş Cd. 20/A, 34379 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 231 00 62	\N	2	\N	\N
237	Pizzeria 14	\N	Şişli, Teşvikiye Mahallesi Güzel Bahçe Sok., Nişantaşı Hancıoğlu Apt, Teşvikiye, D:14/A, 34365 Şişli/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 343 86 60	\N	2	\N	\N
238	Bolpi Pide ve Lahmacun	\N	Şişli, Fulya Mahallesi, Mecidiyeköy, Mehmetçik Cd. 4/A, 34394 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 211 19 00	\N	2	\N	\N
239	Izzetpasa Kokorec	\N	Şişli, İzzet Paşa, Yeni Yol Cd. No:26, 34387 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 224 87 49	\N	2	\N	\N
240	Sarıhan İşkembe Şişli	\N	Şişli, Merkez, Abide-i Hürriyet Cd No: 126, 34400 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 233 93 21	\N	2	\N	\N
241	Nova Plaza Orion Hotel	\N	Nişantaşı, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
242	Swissôtel The Bosphorus Istanbul	\N	Nişantaşı, Vişnezade, Acısu Sk. NO 19, 34357 Beşiktaş/İstanbul, Türkiye	4.8	local_database	\N	\N	(0212) 326 11 00	https://www.swissotel.com/hotels/istanbul/?goto=fiche_hotel&code_hotel=A5D2&merchantid=seo-maps-TR-A5D2&sourceid=aw-cen&utm_medium=seo%20maps&utm_source=google%20Maps&utm_campaign=seo%20maps	2	\N	\N
243	TAKSİM STAR HOTEL BOSPHORUS	\N	Nişantaşı, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
244	CVK Park Bosphorus Hotel Istanbul	\N	Nişantaşı, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
245	Avantgarde Urban Hotel Taksim	\N	Nişantaşı, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
246	dora hotel	\N	Nişantaşı, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
247	Holiday Inn Istanbul Sisli	\N	Nişantaşı, 19 Mayıs, Halaskargazi Cd. No:206, 34360 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 373 38 00	http://www.hisisli.com/	2	\N	\N
248	Ferman Hilal Hotel	\N	Nişantaşı, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
249	The Cettia Istanbul	\N	Nişantaşı, Fulya Mahallesi, Mecidiyeköy, Ortaklar Cd. No:30, 34834 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 912 43 62	https://www.thecettiaistanbul.com/	2	\N	\N
250	Nippon Hotel	\N	Nişantaşı, Kocatepe, Topçu Cd. No:6, 34437 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 313 33 00	http://www.nipponhotel.com.tr/	2	\N	\N
251	Opera Hotel Bosphorus	\N	Nişantaşı, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
252	BellaVista Hostel	\N	Nişantaşı, Bülbül, Turan Cd. No:50, 34400 Beyoğlu/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 361 92 55	\N	2	\N	\N
253	Konak Hotel	\N	Nişantaşı, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
254	Orka Taksim Suites & Hotel	\N	Nişantaşı, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
255	360 Istanbul	\N	Nişantaşı, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
256	5. Kat Restaurant	\N	Nişantaşı, Cihangir, Soğancı Sk. No:3, 34427, 34433 Beyoğlu/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 293 37 74	http://www.5kat.com/	2	\N	\N
257	Mr CAS Hotels	\N	Nişantaşı, Kuloğlu, İstiklal Cd. No:153, 34433 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 00 07	https://www.mrcashotels.com/	2	\N	\N
258	More Restaurant	\N	Nişantaşı, Kocatepe, Lamartin Cd. No:13, 34437 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 361 67 67	https://www.midtown-hotel.com/tr/hotel/more-restaurant/	2	\N	\N
259	Topaz İstanbul	\N	Nişantaşı, Ömer Avni, İnönü Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	4.2	local_database	\N	\N	0531 329 41 11	http://www.topazistanbul.com/	2	\N	\N
260	Tarihi Cumhuriyet Meyhanesi	\N	Nişantaşı, Kamer Hatun Sahne Sokak, Hüseyinağa, Kamer Hatun Cd. No:27, 34435 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 252 08 86	http://www.tarihicumhuriyetmeyhanesi.com.tr/	2	\N	\N
261	Avantgarde Urban Hotel Levent	\N	Etiler, Esentepe, Büyükdere Cd. No:161, 34394 Şişli/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 337 04 44	https://www.avantgardecollection.com/urban-levent-hotel	2	\N	\N
262	Sunset Grill & Bar	\N	Etiler, Kuruçeşme Mahallesi Ulus Park, Kuruçeşme, Yol Sokağı No:2, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 287 03 57	https://www.sunsetgrillbar.com/	2	\N	\N
263	Mangerie	\N	Etiler, Bebek, Cevdet Paşa Cd. No:69, 34342 Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 263 51 99	http://www.mangeriebebek.com/	2	\N	\N
264	The House Cafe	\N	Etiler, Levent, Kanyon AVM No:185, 34394 Beşiktaş/İstanbul, Türkiye	3.3	local_database	\N	\N	(0212) 353 53 75	\N	2	\N	\N
265	Vitrin Meyhane Etiler	\N	Etiler, Nispetiye Mahallesi, Aytar Cd. 3/A, 34443 Etiler/Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 232 20 46	\N	2	\N	\N
266	Develi Etiler	\N	Etiler, Etiler, Tepecik Yolu No:22 D:22, 34337 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 263 25 71	http://www.develikebap.com/	2	\N	\N
267	Paper Moon	\N	Etiler, Etiler, Ahmet Adnan Saygun Cd., 34340 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 282 16 16	https://www.papermoonrestaurants.com/paper-moon-istanbul.html	2	\N	\N
268	Midpoint Bebek	\N	Etiler, Bebek, Cevdet Paşa Cd. No: 35, 34342 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	0530 918 00 26	https://www.midpoint.com.tr/	2	\N	\N
269	Ulus Cafe	\N	Etiler, Kuruçeşme Adnan Saygun Cad, Kuruçeşme, Ulus Parkı 71/B, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 263 84 37	http://uluscafe.com/	2	\N	\N
270	Köfteci Ramiz	\N	Etiler, Levent, Büyükdere Cd. No:224, 34330 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 344 04 31	https://www.google.com/url?sa=D&q=http://www.kofteciramiz.com/	2	\N	\N
271	Sevgi simit (Yücel Simit)	\N	Etiler, Nisbetiye, Nisbetiye Cd No:32, 34340 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 270 21 51	\N	2	\N	\N
272	Karafırın Ulus	\N	Etiler, Kültür, Ahmet Adnan Saygun Cd. 39/A, 34340 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 265 21 51	https://www.karafirin.com.tr/	2	\N	\N
273	Bolulu Hasan Usta Süt Tatlıları	\N	Etiler, Nisbetiye, Nisbetiye Cd No:28/11, 34340 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 264 16 64	http://www.bhu.com.tr/	2	\N	\N
274	Galatasaraylılar Derneği Lokali	\N	Etiler, Levent, 34335 Beşiktaş/İstanbul, Türkiye	4.5	local_database	\N	\N	0538 812 18 68	https://galatasaraylilardernegi.org.tr/	2	\N	\N
275	Naturel Park Spor Tesisleri	\N	Etiler, Akatlar Mahallesi, Etiler, Ebulula Mardin Cd. No:2/1, 34335 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 352 70 65	https://www.instagram.com/etilernaturelpark	2	\N	\N
276	Gunaydin	\N	Etiler, Etiler, Nisbetiye Cd No:104 D:F, 34337 Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 282 01 53	https://www.gunaydinkoftedoner.com/	2	\N	\N
277	Diet Club	\N	Etiler, Levent, Hanımeli Sk. No:5, 34330 Beşiktaş/İstanbul, Türkiye	1.9	local_database	\N	\N	(0212) 270 66 11	\N	2	\N	\N
278	BURGERLAB	\N	Etiler, Food Court, Kültür, Ak Merkez AVM, 34337 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 282 18 18	\N	2	\N	\N
279	McDonald's İstanbul Özdilek Avm	\N	Etiler, Esentepe Büyükdere Caddesi Özdilek AVM, Esentepe, D:177/43, 34394 Şişli/İstanbul, Türkiye	2.7	local_database	\N	\N	0533 281 00 30	https://www.mcdonalds.com.tr/kurumsal/restoranlar	2	\N	\N
280	Ulubey Manti	\N	Etiler, 120 A, Etiler, Nisbetiye Cd, 34337 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 265 01 57	\N	2	\N	\N
281	Avantgarde Urban Hotel Levent	\N	Levent, Esentepe, Büyükdere Cd. No:161, 34394 Şişli/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 337 04 44	https://www.avantgardecollection.com/urban-levent-hotel	2	\N	\N
282	The House Cafe	\N	Levent, Levent, Kanyon AVM No:185, 34394 Beşiktaş/İstanbul, Türkiye	3.3	local_database	\N	\N	(0212) 353 53 75	\N	2	\N	\N
283	Vitrin Meyhane Etiler	\N	Levent, Nispetiye Mahallesi, Aytar Cd. 3/A, 34443 Etiler/Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 232 20 46	\N	2	\N	\N
284	Paper Moon	\N	Levent, Etiler, Ahmet Adnan Saygun Cd., 34340 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 282 16 16	https://www.papermoonrestaurants.com/paper-moon-istanbul.html	2	\N	\N
285	Develi Etiler	\N	Levent, Etiler, Tepecik Yolu No:22 D:22, 34337 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 263 25 71	http://www.develikebap.com/	2	\N	\N
286	Burger King	\N	Levent, Fulya, Prof. Dr. Bülent Tarcan Sk. No:86, 34394 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
287	Köfteci Ramiz	\N	Levent, Levent, Büyükdere Cd. No:224, 34330 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 344 04 31	https://www.google.com/url?sa=D&q=http://www.kofteciramiz.com/	2	\N	\N
288	Kayra Life Restaurant	\N	Levent, Konaklar, Akasyalı Sk. No:3 No:3, 34330 Beşiktaş/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 279 27 10	\N	2	\N	\N
289	Sevgi simit (Yücel Simit)	\N	Levent, Nisbetiye, Nisbetiye Cd No:32, 34340 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 270 21 51	\N	2	\N	\N
290	Galatasaraylılar Derneği Lokali	\N	Levent, Levent, 34335 Beşiktaş/İstanbul, Türkiye	4.5	local_database	\N	\N	0538 812 18 68	https://galatasaraylilardernegi.org.tr/	2	\N	\N
291	Bolulu Hasan Usta Süt Tatlıları	\N	Levent, Nisbetiye, Nisbetiye Cd No:28/11, 34340 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 264 16 64	http://www.bhu.com.tr/	2	\N	\N
292	Naturel Park Spor Tesisleri	\N	Levent, Akatlar Mahallesi, Etiler, Ebulula Mardin Cd. No:2/1, 34335 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 352 70 65	https://www.instagram.com/etilernaturelpark	2	\N	\N
293	McDonald's İstanbul Özdilek Avm	\N	Levent, Esentepe Büyükdere Caddesi Özdilek AVM, Esentepe, D:177/43, 34394 Şişli/İstanbul, Türkiye	2.7	local_database	\N	\N	0533 281 00 30	https://www.mcdonalds.com.tr/kurumsal/restoranlar	2	\N	\N
294	Hışım Lezzet Lokantası	\N	Levent, Sultan Selim, Turan Sokağı No:4, 34415 Kağıthane/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 281 72 32	http://www.hisimborek.com.tr/	2	\N	\N
295	Diet Club	\N	Levent, Levent, Hanımeli Sk. No:5, 34330 Beşiktaş/İstanbul, Türkiye	1.9	local_database	\N	\N	(0212) 270 66 11	\N	2	\N	\N
296	Yesillik	\N	Levent, Levent, Büyükdere Cd. Metrocity Alışveriş Merkezi No:171 D:252, 34394 Beşiktaş/Istanbul - Europe, Türkiye	\N	local_database	\N	\N	(0212) 344 00 04	\N	2	\N	\N
297	Karanlıkta Yemek	\N	Levent, Sultan Selim, Akyol Sitesi Çk No:10, 34415 Kağıthane/İstanbul, Türkiye	4.5	local_database	\N	\N	0532 342 25 38	http://www.karanliktayemek.com/	2	\N	\N
298	Pilove	\N	Levent, Esentepe, Eski Büyükdere Cd. 171/237, 34330 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 344 06 00	\N	2	\N	\N
299	Konak	\N	Levent, Levent, Büyükdere Cd. No:2, 34330 Beşiktaş/İstanbul, Türkiye	2.4	local_database	\N	\N	(0212) 280 85 61	\N	2	\N	\N
300	Le Pain Quotidien	\N	Levent, Esentepe, Kanyon AVM No:185, 34330 Şişli/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 450 50 20	https://lepainquotidien.com/tr/stores/kanyon%20mall?utm_source=gmb&utm_medium=business-listing	2	\N	\N
301	Orient Express & Spa by Orka Hotels	\N	Sultanahmet, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
302	GLK PREMIER Regency Suites & Spa	\N	Sultanahmet, Cankurtaran, Akbıyık Cd. No:46, 34122 Fatih/İstanbul, Türkiye	4.6	local_database	\N	\N	0530 568 25 68	https://www.regencysuitesistanbul.com/	2	\N	\N
303	Gülhanepark Hotel & Spa	\N	Sultanahmet, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
304	Hotel Spectra	\N	Sultanahmet, Binbirdirek, Şht. Mehmetpaşa Ykş. No:2, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 516 35 46	http://www.hotelspectra.com/	2	\N	\N
305	Legacy Ottoman Hotel	\N	Sultanahmet, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
306	GLK PREMIER The Home Suites & Spa	\N	Sultanahmet, Küçük Ayasofya, Küçük Ayasofya Cd. No:60, 34122 Fatih/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 638 88 01	http://www.thehomesuites.com/	2	\N	\N
307	Orka Royal Hotel	\N	Sultanahmet, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
308	Pierre Loti Hotel	\N	Sultanahmet, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
309	Antik Hotel Istanbul	\N	Sultanahmet, Mimar Kemalettin Mahallesi, Beyazıt, Sekbanbaşı Sk. No:6, 34130 Fatih/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 638 58 58	http://www.antikhotel.com/	2	\N	\N
310	Blue House Hotel & Rooftop – 360 Sunset View of Istanbul	\N	Sultanahmet, Cankurtaran, Dalbastı Sk. No:14, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 638 90 10	https://www.bluehouseistanbul.com/	2	\N	\N
311	Viva Deluxe Hotel	\N	Sultanahmet, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
312	Matbah Restaurant	\N	Sultanahmet, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
313	Tria Hotel Istanbul	\N	Sultanahmet, Cankurtaran, Terbıyık Sk. No:7, 34122 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 45 18	http://www.triahotelistanbul.com/	2	\N	\N
314	Prestige Hotel Istanbul Laleli	\N	Sultanahmet, Mimar Kemalettin, Koska Cd. No: 6, 34130 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 82 80	http://www.hotelprestige.com.tr/	2	\N	\N
315	Yasmak Sultan Hotel	\N	Sultanahmet, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
316	Istanbul Town Hotel	\N	Sultanahmet, Mimar Kemalettin, Kalaycı Şevki Sk. No:4, 34126 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 458 00 27	https://www.istanbultownhotel.com/	2	\N	\N
317	Akgun Hotel Beyazit	\N	Sultanahmet, Mimar Kemalettin, Haznedar Sk. No:6, 34490 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 638 28 38	http://www.akgunotel.com.tr/	2	\N	\N
318	Darkhill Hotel	\N	Sultanahmet, Mimar Kemalettin, Koca Ragıppaşa Cd No:9, 34130 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 638 89 00	http://www.darkhillhotel.com/	2	\N	\N
319	Hotel Zurich Istanbul	\N	Sultanahmet, Balabanağa, Vidinli Tevfikpaşa Cd No:14, 34134 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 512 23 50	http://www.hotelzurichistanbul.com/	2	\N	\N
320	GLK PREMIER Acropol Suites	\N	Sultanahmet, Sultanahmet (Old City, Cankurtaran, Akbıyık Cd. No:21, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 638 90 21	https://www.acropolhotel.com/	2	\N	\N
321	Georges Hotel Galata	\N	Beyoğlu, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
322	Nova Plaza Orion Hotel	\N	Beyoğlu, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
323	TAKSİM STAR HOTEL BOSPHORUS	\N	Beyoğlu, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
324	Ramada By Wyndham Istanbul Pera	\N	Beyoğlu, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
325	CVK Park Bosphorus Hotel Istanbul	\N	Beyoğlu, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
326	Avantgarde Urban Hotel Taksim	\N	Beyoğlu, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
327	dora hotel	\N	Beyoğlu, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
328	Ferman Hilal Hotel	\N	Beyoğlu, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
329	Anemon Koleksiyon Galata Otel	\N	Beyoğlu, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
330	360 Istanbul	\N	Beyoğlu, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
331	Orka Taksim Suites & Hotel	\N	Beyoğlu, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
332	Nippon Hotel	\N	Beyoğlu, Kocatepe, Topçu Cd. No:6, 34437 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 313 33 00	http://www.nipponhotel.com.tr/	2	\N	\N
333	BellaVista Hostel	\N	Beyoğlu, Bülbül, Turan Cd. No:50, 34400 Beyoğlu/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 361 92 55	\N	2	\N	\N
334	Opera Hotel Bosphorus	\N	Beyoğlu, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
335	Buyuk Londra Hotel's Terrace Bar	\N	Beyoğlu, Asmalı Mescit, Meşrutiyet Cd. No:53, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 245 06 70	https://londrahotel.net/tr/	2	\N	\N
336	Konak Hotel	\N	Beyoğlu, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
337	Mr CAS Hotels	\N	Beyoğlu, Kuloğlu, İstiklal Cd. No:153, 34433 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 00 07	https://www.mrcashotels.com/	2	\N	\N
338	Van Kahvaltı Evi	\N	Beyoğlu, Kılıçali Paşa, Defterdar Ykş. 52/A, 34425 Beyoğlu/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 293 64 37	\N	2	\N	\N
339	5. Kat Restaurant	\N	Beyoğlu, Cihangir, Soğancı Sk. No:3, 34427, 34433 Beyoğlu/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 293 37 74	http://www.5kat.com/	2	\N	\N
340	Mikla	\N	Beyoğlu, The Marmara Pera, Asmalı Mescit, Meşrutiyet Cd. No:15, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 56 56	http://www.miklarestaurant.com/	2	\N	\N
341	Orient Express & Spa by Orka Hotels	\N	Galata, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
342	Georges Hotel Galata	\N	Galata, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
343	Gülhanepark Hotel & Spa	\N	Galata, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
344	Legacy Ottoman Hotel	\N	Galata, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
345	Orka Royal Hotel	\N	Galata, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
346	TAKSİM STAR HOTEL BOSPHORUS	\N	Galata, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
347	Viva Deluxe Hotel	\N	Galata, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
348	Nova Plaza Orion Hotel	\N	Galata, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
349	Ramada By Wyndham Istanbul Pera	\N	Galata, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
350	Yasmak Sultan Hotel	\N	Galata, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
351	CVK Park Bosphorus Hotel Istanbul	\N	Galata, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
352	Anemon Koleksiyon Galata Otel	\N	Galata, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
353	Matbah Restaurant	\N	Galata, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
354	360 Istanbul	\N	Galata, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
355	Pierre Loti Hotel	\N	Galata, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
356	Avantgarde Urban Hotel Taksim	\N	Galata, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
357	Orka Taksim Suites & Hotel	\N	Galata, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
358	Ferman Hilal Hotel	\N	Galata, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
359	Hotel Vicenza	\N	Galata, Kemal Paşa, Fevziye Cd. No:3, 34134 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 520 17 01	http://www.hotelvicenzaistanbul.com/	2	\N	\N
360	Golden Horn Hotel	\N	Galata, Hoca Paşa, Ebussuud Cd. No:26, 34110 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 74 74	http://www.thegoldenhorn.com/	2	\N	\N
361	Orient Express & Spa by Orka Hotels	\N	Karaköy, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
362	Georges Hotel Galata	\N	Karaköy, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
363	Gülhanepark Hotel & Spa	\N	Karaköy, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
364	Legacy Ottoman Hotel	\N	Karaköy, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
365	Orka Royal Hotel	\N	Karaköy, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
366	TAKSİM STAR HOTEL BOSPHORUS	\N	Karaköy, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
367	Nova Plaza Orion Hotel	\N	Karaköy, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
368	Viva Deluxe Hotel	\N	Karaköy, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
369	Ramada By Wyndham Istanbul Pera	\N	Karaköy, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
370	CVK Park Bosphorus Hotel Istanbul	\N	Karaköy, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
371	Yasmak Sultan Hotel	\N	Karaköy, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
372	Anemon Koleksiyon Galata Otel	\N	Karaköy, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
373	Matbah Restaurant	\N	Karaköy, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
374	Avantgarde Urban Hotel Taksim	\N	Karaköy, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
375	360 Istanbul	\N	Karaköy, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
376	Pierre Loti Hotel	\N	Karaköy, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
377	Orka Taksim Suites & Hotel	\N	Karaköy, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
378	Ferman Hilal Hotel	\N	Karaköy, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
379	Golden Horn Hotel	\N	Karaköy, Hoca Paşa, Ebussuud Cd. No:26, 34110 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 74 74	http://www.thegoldenhorn.com/	2	\N	\N
380	Buyuk Londra Hotel's Terrace Bar	\N	Karaköy, Asmalı Mescit, Meşrutiyet Cd. No:53, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 245 06 70	https://londrahotel.net/tr/	2	\N	\N
381	Swissôtel The Bosphorus Istanbul	\N	Beşiktaş, Vişnezade, Acısu Sk. NO 19, 34357 Beşiktaş/İstanbul, Türkiye	4.8	local_database	\N	\N	(0212) 326 11 00	https://www.swissotel.com/hotels/istanbul/?goto=fiche_hotel&code_hotel=A5D2&merchantid=seo-maps-TR-A5D2&sourceid=aw-cen&utm_medium=seo%20maps&utm_source=google%20Maps&utm_campaign=seo%20maps	2	\N	\N
382	Symbola Bosphorus Istanbul	\N	Beşiktaş, Ortaköy, Portakal Ykş. Cd. No:17, 34347 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	0533 136 37 67	http://www.symbolabosphorus.com/	2	\N	\N
383	CVK Park Bosphorus Hotel Istanbul	\N	Beşiktaş, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
384	Opera Hotel Bosphorus	\N	Beşiktaş, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
385	Feriye Lokantasi	\N	Beşiktaş, Yıldız, Çırağan Cd. No:44, 34347 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 227 22 17	http://www.feriye.com/	2	\N	\N
386	The House Cafe	\N	Beşiktaş, Yıldız, Ortaköy Salhanesi Sk. No:1, 34349 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 227 26 99	\N	2	\N	\N
387	Mado Ortaköy	\N	Beşiktaş, Mecidiye Mahallesi, Ortaköy, İskele Sk. No:24, 34347 Beşiktaş/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 259 20 00	https://www.mado.com.tr/	2	\N	\N
388	Konak Hotel	\N	Beşiktaş, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
389	İBB Fethipaşa Sosyal Tesisleri	\N	Beşiktaş, Kuzguncuk Mahallesi Paşa Limanı Caddesi Nacak Sokak No:6 Kapı No: 14/2, Kuzguncuk, 34674 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	444 1 034	https://tesislerimiz.ibb.istanbul/fethipasa-sosyal-tesisi/	2	\N	\N
390	Topaz İstanbul	\N	Beşiktaş, Ömer Avni, İnönü Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	4.2	local_database	\N	\N	0531 329 41 11	http://www.topazistanbul.com/	2	\N	\N
391	ORTAKÖY POLİSEVİ SOSYAL TESİSİ (‼️KONAKLAMA YOKTUR‼️)	\N	Beşiktaş, Ortaköy, Portakal Ykş. Cd. No:58, 34347 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	0505 318 34 76	\N	2	\N	\N
392	Nişantaşı Başköşe	\N	Beşiktaş, Harbiye, Bronz Sk. No:5/1, 34370 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 230 38 68	http://www.nisantasibaskose.com/	2	\N	\N
393	The Brasserie Restaurant	\N	Beşiktaş, 1, Gümüşsuyu, Asker Ocağı Cd., 34437 Beyoğlu/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 368 44 44	http://istanbul.intercontinental.com/	2	\N	\N
394	Ali Baba Iskender Kebapcisi	\N	Beşiktaş, Sinanpaşa, Yeni Hamam Sk. No:11, 34353 Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 227 77 28	http://www.alibabaiskender.com/	2	\N	\N
395	Elma Cafe & Pub	\N	Beşiktaş, Sinanpaşa Köyiçi Cad, Sinanpaşa, Leşker Sk. No:1, 34353 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 261 36 34	http://www.elmacafepub.com/	2	\N	\N
396	Murat Muhallebi Beşiktaş Şube	\N	Beşiktaş, Sinanpaşa, Ihlamurdere Cd. No:10, 34353 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 258 45 20	\N	2	\N	\N
397	Renkli Limon	\N	Beşiktaş, Sinanpaşa, Ihlamurdere Caddesi. Alaybeyi Sok. No:20, 34353 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 258 80 85	\N	2	\N	\N
398	Dragon Restaurant	\N	Beşiktaş, Harbiye, Cumhuriyet Cd. No:50, 34367 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 231 62 00	http://dragonrestaurant.com.tr/	2	\N	\N
399	Okka	\N	Beşiktaş, Akaretler, Vişnezade, Süleyman Seba Cd. No: 22, 34357 Beşiktaş/İstanbul, Türkiye	5	local_database	\N	\N	(0212) 381 21 21	\N	2	\N	\N
400	Sharap	\N	Beşiktaş, Vişnezade, Süleyman Seba Cd. Bjk Plaza A Blok 1 No:48, 34357 Beşiktaş/İstanbul, Türkiye	5	local_database	\N	\N	(0212) 236 93 55	http://193.255.140.26/	2	\N	\N
401	Symbola Bosphorus Istanbul	\N	Ortaköy, Ortaköy, Portakal Ykş. Cd. No:17, 34347 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	0533 136 37 67	http://www.symbolabosphorus.com/	2	\N	\N
402	Avantgarde Urban Hotel Levent	\N	Ortaköy, Esentepe, Büyükdere Cd. No:161, 34394 Şişli/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 337 04 44	https://www.avantgardecollection.com/urban-levent-hotel	2	\N	\N
403	Feriye Lokantasi	\N	Ortaköy, Yıldız, Çırağan Cd. No:44, 34347 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 227 22 17	http://www.feriye.com/	2	\N	\N
404	Sunset Grill & Bar	\N	Ortaköy, Kuruçeşme Mahallesi Ulus Park, Kuruçeşme, Yol Sokağı No:2, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 287 03 57	https://www.sunsetgrillbar.com/	2	\N	\N
405	The House Cafe	\N	Ortaköy, Yıldız, Ortaköy Salhanesi Sk. No:1, 34349 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 227 26 99	\N	2	\N	\N
406	ORTAKÖY POLİSEVİ SOSYAL TESİSİ (‼️KONAKLAMA YOKTUR‼️)	\N	Ortaköy, Ortaköy, Portakal Ykş. Cd. No:58, 34347 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	0505 318 34 76	\N	2	\N	\N
407	Mado Ortaköy	\N	Ortaköy, Mecidiye Mahallesi, Ortaköy, İskele Sk. No:24, 34347 Beşiktaş/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 259 20 00	https://www.mado.com.tr/	2	\N	\N
408	HuQQa Kuruçeşme	\N	Ortaköy, Kuruçeşme, Muallim Naci Cd. No:56, 34345 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 265 06 66	https://www.huqqa.com/tr/kurucesme-huqqa/	2	\N	\N
409	Ulus Cafe	\N	Ortaköy, Kuruçeşme Adnan Saygun Cad, Kuruçeşme, Ulus Parkı 71/B, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 263 84 37	http://uluscafe.com/	2	\N	\N
410	Galatasaray İsland	\N	Ortaköy, Kuruçeşme, Galatasaray Adası, 34000 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	\N	\N	2	\N	\N
411	MEŞHUR BURSA İNEGÖL KÖFTECiSi	\N	Ortaköy, Mecidiye, Dereboyu Cd. No:49A, 34347 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 258 13 12	https://www.tripadvisor.com.tr/Restaurants-g2540453-Inegol.html	2	\N	\N
412	Meshur Adiyaman Cig Koftecisi	\N	Ortaköy, Ortaköy, Dereboyu Cd. No:102 D:b, 34347 Beşiktaş/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 259 96 94	http://www.adiyamancigkoftecisi.com/	2	\N	\N
413	Chinese & Sushi Express	\N	Ortaköy, Ortaköy Ambarlıdere Cad Lotus World No: 6, Ortaköy, D:1, 34400 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	444 1 246	https://www.sushiexpress.com.tr/tr/subeler/lotus-ortakoy	2	\N	\N
414	inari Omakase Kuruçeşme	\N	Ortaköy, Kuruçeşme, Kuruçeşme Cd. No:11, 34345 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 265 96 01	https://www.inariomakase.com/kurucesme/	2	\N	\N
415	Ay Balik	\N	Ortaköy, Kuruçeşme, Kuruçeşme Cd. No:15, 34345 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	0532 417 75 71	\N	2	\N	\N
416	Golden Kokoreç	\N	Ortaköy, Mecidiye, Dereboyu Cd. No:3, 34347 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 259 79 89	\N	2	\N	\N
417	Ali Usta Burma Kadayif & Baklava	\N	Ortaköy, Mecidiye, Dereboyu Cd. No:135A, 34347 Beşiktaş/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 227 41 90	http://www.aliustabaklava.com/	2	\N	\N
418	Baklavaci Dedeoglu	\N	Ortaköy, Ortaköy, Dereboyu Cd. 61/A, 34347 Beşiktaş/İstanbul, Türkiye	4.7	local_database	\N	\N	(0212) 259 63 10	\N	2	\N	\N
419	Bizce Ortaköy Kuru Fasulyecisi	\N	Ortaköy, Ortaköy, Dereboyu Cd. No:52, 34347 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 236 06 21	https://ortakoykurufasulyecisi.com/	2	\N	\N
420	Dedeoğlu Baklava -Balmumcu	\N	Ortaköy, Balmumcu Mahallesi Mustafa İzzet Efendi sokak no :8, Mecidiye, 34349 Beşiktaş/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 288 78 78	http://www.dedeoglu.com.tr/	2	\N	\N
421	Sarnic Pub & Hotel	\N	Kadıköy, Caferağa, Dumlupınar Sk. No:12, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	0533 729 09 30	http://www.sarnicmeyhaneveotel.com/	2	\N	\N
422	Moda Sea Club 1935	\N	Kadıköy, Moda Caddesi, Caferağa, Ferit Tek Sok. No:1, 34710 Kadıköy/İstanbul, Türkiye	4.5	local_database	\N	\N	(0216) 346 90 72	http://www.modadenizkulubu.org.tr/	2	\N	\N
423	Hacioglu	\N	Kadıköy, Caferağa, Yasa Cd. No:40/A, 34710 Kadıköy/İstanbul, Türkiye	3	local_database	\N	\N	(0216) 414 44 14	https://www.hacioglu.com.tr/	2	\N	\N
424	Görgülü Pastane ve Restaurant	\N	Kadıköy, Acıbadem, Acıbadem Cd. No:94, 34718 Kadıköy/İstanbul, Türkiye	4.1	local_database	\N	\N	(0216) 545 09 02	http://www.gorgulupasta.com/	2	\N	\N
425	Harput Koz Kebap Salonu	\N	Kadıköy, Hasanpaşa, Ulusuluk Sk. No:2, 34722 Kadıköy/İstanbul, Türkiye	3.4	local_database	\N	\N	(0216) 326 71 53	\N	2	\N	\N
426	Mercan Kokoreç	\N	Kadıköy, Caferağa, Muvakkıthane Cd. 15/A, 34710 Kadıköy/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 450 56 56	http://www.mercankokorec.com.tr/	2	\N	\N
427	Pide Sun	\N	Kadıköy, Şükran Apt, Caferağa, Moda Cd. 67/B, 34710 Kadıköy/İstanbul, Türkiye	4.2	local_database	\N	\N	(0216) 450 43 43	http://www.pidesun.com/servis.htm	2	\N	\N
428	Pilav Station	\N	Kadıköy, Caferağa, Piri Çavuş Sk. No:37 D:B, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	(0216) 348 52 56	\N	2	\N	\N
429	Hosta	\N	Kadıköy, Osmanağa, Söğütlü Çeşme Cd No:6, 34714 Kadıköy/İstanbul, Türkiye	2.8	local_database	\N	\N	(0216) 405 16 78	http://www.hosta.com.tr/	2	\N	\N
430	Doyuran Kokoreç	\N	Kadıköy, Rasimpaşa, Rıhtım Cd. No:24, 34716 Kadıköy/İstanbul, Türkiye	3.7	local_database	\N	\N	(0216) 330 47 00	\N	2	\N	\N
431	Karafırın Hasanpaşa	\N	Kadıköy, 13, Hasanpaşa, Fahrettin Kerim Gökay Cd, 34722 Kadıköy/İstanbul, Türkiye	3.4	local_database	\N	\N	(0216) 414 96 50	https://www.karafirin.com.tr/	2	\N	\N
432	Kadı Lokantası	\N	Kadıköy, Osmanağa, Halitağa Cd. No:44, 34714 Kadıköy/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 449 27 17	\N	2	\N	\N
433	Fil Shot	\N	Kadıköy, Caferağa, Sakız Gülü Sk. No:26/A, 34710 Kadıköy/İstanbul, Türkiye	4.1	local_database	\N	\N	0532 651 95 80	http://filbistro.com/	2	\N	\N
434	Hacı Ali 2 Et Lokantası	\N	Kadıköy, Osmanağa, Karadut Sk. No:5, 34714 Kadıköy/İstanbul, Türkiye	4.4	local_database	\N	\N	(0216) 346 32 81	\N	2	\N	\N
435	Kimyon Kadıköy	\N	Kadıköy, Caferağa, Kadife Sk. No:17 D:C, 34710 Kadıköy/İstanbul, Türkiye	4.3	local_database	\N	\N	(0216) 330 48 45	http://www.kimyon.com.tr/	2	\N	\N
436	Burger King - Acıbadem	\N	Kadıköy, Acıbadem, Dar Sk. No:1/1, 34718 Kadıköy/İstanbul, Türkiye	3.4	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
437	Tek Buffet	\N	Kadıköy, Caferağa, Moda Cd. No:70 D:70, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	(0216) 338 17 70	\N	2	\N	\N
438	Pilav Dunyasi	\N	Kadıköy, Caferağa, Sakız Gülü Sk. 21/2, 34710 Kadıköy/İstanbul, Türkiye	2.8	local_database	\N	\N	(0216) 348 50 87	\N	2	\N	\N
439	Çiya Sofrası	\N	Kadıköy, Caferağa, Güneşli Bahçe Sok, 34710 Kadıköy/İstanbul, Türkiye	4	local_database	\N	\N	(0216) 418 51 15	http://www.ciya.com.tr/	2	\N	\N
440	Cafe & Shop Kadıköy	\N	Kadıköy, Caferağa Mh Bahariye Cad, Caferağa, Hacı Şükrü Sk. No:11, 34710 Kadıköy/İstanbul, Türkiye	4.4	local_database	\N	\N	(0216) 418 60 64	\N	2	\N	\N
441	İBB Fethipaşa Sosyal Tesisleri	\N	Üsküdar, Kuzguncuk Mahallesi Paşa Limanı Caddesi Nacak Sokak No:6 Kapı No: 14/2, Kuzguncuk, 34674 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	444 1 034	https://tesislerimiz.ibb.istanbul/fethipasa-sosyal-tesisi/	2	\N	\N
442	Ozsut	\N	Üsküdar, CAPİTOL AVM Altunizade Mahallesi PROF.DR.MAHİR İZ CAD. NO:7 ZEMİN KAT, Altunizade, 34662 Üsküdar/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 651 41 94	http://www.ozsut.com.tr/	2	\N	\N
443	Cafe 5. Cadde	\N	Üsküdar, Salacak, Üsküdar Harem Sahil Yolu No:29, 34000 Üsküdar/İstanbul, Türkiye	3.9	local_database	\N	\N	\N	\N	2	\N	\N
444	Üsküdar Pide	\N	Üsküdar, Valide-i Atik, Çavuşdere Cd. No:86, 34672 Üsküdar/İstanbul, Türkiye	3.8	local_database	\N	\N	(0216) 492 72 72	http://www.meshuristanbulpide.com/	2	\N	\N
445	ÜSKÜDAR DÖNER LAHMACUN PİDE SALONU	\N	Üsküdar, Valide-i Atik, Çavuşdere Cd. 72/A, 34664 Üsküdar/İstanbul, Türkiye	4.2	local_database	\N	\N	(0216) 341 24 98	\N	2	\N	\N
446	Temel Reis Akcaabat Koftecisi	\N	Üsküdar, Kerem Yilmazer Sahnesi, Valide-i Atik, Dr. Fahri Atabey Cd. No:79 Alti, 34668 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	(0216) 553 42 20	http://kuymakadam.com/	2	\N	\N
447	Şişmanoğlu Tantuni	\N	Üsküdar, Ahmediye, Dr. Fahri Atabey Cd., 34672 Üsküdar/İstanbul, Türkiye	4.2	local_database	\N	\N	(0216) 343 23 20	http://sismanoglutantuni.com/	2	\N	\N
448	As Kebap	\N	Üsküdar, Sultantepe, 34664 Üsküdar/İstanbul, Türkiye	3.9	local_database	\N	\N	(0216) 334 21 24	\N	2	\N	\N
449	Limon Pide Kebap Evi	\N	Üsküdar, Hacı Hesna Hatun Mh, Selmanağa Bostanı Sk. No:8, 34777 Üsküdar/İstanbul, Türkiye	3.6	local_database	\N	\N	(0216) 553 43 43	https://www.limonpidekebapevi.com/	2	\N	\N
450	Altın Şiş Restaurant	\N	Üsküdar, Ahmediye, Kefçedede Mektebi Sokağı No:10, 34672 Üsküdar/İstanbul, Türkiye	4.3	local_database	\N	\N	(0216) 553 23 70	http://altınşiş.com.tr/	2	\N	\N
451	Genc Kebap	\N	Üsküdar, Mimar Sinan, Selmani Pak Cd. No:1, 34674 Üsküdar/İstanbul, Türkiye	3.3	local_database	\N	\N	(0216) 391 96 63	\N	2	\N	\N
452	Ozbolu Lokantasi	\N	Üsküdar, Mimar Sinan, Atlas Sk. No:11, 34672 Üsküdar/İstanbul, Türkiye	4.1	local_database	\N	\N	(0216) 532 73 31	http://ozbolu-lokantasi.com/	2	\N	\N
453	Güneş Kokoreç	\N	Üsküdar, Mimar Sinan, Balaban Cd. No:33, 34672 Üsküdar/İstanbul, Türkiye	4.2	local_database	\N	\N	0541 842 19 37	https://www.instagram.com/guneskokorec1974?igsh=eDRnYWxwajBzMTg4	2	\N	\N
454	Pilav House	\N	Üsküdar, Mimar Sinan, Selami Ali Efendi Cad. 31/A, 34672 Üsküdar/Istanbul - Asia, Türkiye	3.9	local_database	\N	\N	(0216) 391 29 54	\N	2	\N	\N
455	Çiğ Köfteci Ömer Usta	\N	Üsküdar, Mimar Sinan, Selmani Pak Cd. No:32 D:E, 34664 Üsküdar/İstanbul, Türkiye	3.7	local_database	\N	\N	\N	https://cigkofteciomerusta.com/	2	\N	\N
456	Üsküdarlı Beyzade	\N	Üsküdar, Ahmediye, Dr. Fahri Atabey Cd. No:24, 34672 Üsküdar/İstanbul, Türkiye	4.4	local_database	\N	\N	\N	\N	2	\N	\N
457	Uluç Ekmek Arası	\N	Üsküdar, Aziz Mahmut Hüdayi, Hakimiyeti Milliye Cd. 58 A, 34672 Üsküdar/İstanbul, Türkiye	4.5	local_database	\N	\N	(0216) 310 77 88	\N	2	\N	\N
458	Burger King - Üsküdar Meydan	\N	Üsküdar, Aziz Mahmut Hüdayi, Hakimiyeti Milliye Cd. No:78, 34672 Üsküdar/İstanbul, Türkiye	3.2	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
459	Hacıoğlu	\N	Üsküdar, Aziz Mahmut Hüdayi, Hakimiyeti Milliye Cd. No:46, 34672 Üsküdar/İstanbul, Türkiye	2.8	local_database	\N	\N	(0216) 391 31 01	https://www.hacioglu.com.tr/	2	\N	\N
460	Beyaz Saray	\N	Üsküdar, Mimar Sinan, Hakimiyeti Milliye Cd. No:22, 34672 Üsküdar/İstanbul, Türkiye	3.1	local_database	\N	\N	(0216) 341 25 08	\N	2	\N	\N
461	Orient Express & Spa by Orka Hotels	\N	Eminönü, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
462	Georges Hotel Galata	\N	Eminönü, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
463	Gülhanepark Hotel & Spa	\N	Eminönü, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
464	GLK PREMIER Regency Suites & Spa	\N	Eminönü, Cankurtaran, Akbıyık Cd. No:46, 34122 Fatih/İstanbul, Türkiye	4.6	local_database	\N	\N	0530 568 25 68	https://www.regencysuitesistanbul.com/	2	\N	\N
465	Hotel Spectra	\N	Eminönü, Binbirdirek, Şht. Mehmetpaşa Ykş. No:2, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 516 35 46	http://www.hotelspectra.com/	2	\N	\N
466	Legacy Ottoman Hotel	\N	Eminönü, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
467	Orka Royal Hotel	\N	Eminönü, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
468	Antik Hotel Istanbul	\N	Eminönü, Mimar Kemalettin Mahallesi, Beyazıt, Sekbanbaşı Sk. No:6, 34130 Fatih/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 638 58 58	http://www.antikhotel.com/	2	\N	\N
469	Pierre Loti Hotel	\N	Eminönü, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
470	GLK PREMIER The Home Suites & Spa	\N	Eminönü, Küçük Ayasofya, Küçük Ayasofya Cd. No:60, 34122 Fatih/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 638 88 01	http://www.thehomesuites.com/	2	\N	\N
471	Prestige Hotel Istanbul Laleli	\N	Eminönü, Mimar Kemalettin, Koska Cd. No: 6, 34130 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 82 80	http://www.hotelprestige.com.tr/	2	\N	\N
472	Blue House Hotel & Rooftop – 360 Sunset View of Istanbul	\N	Eminönü, Cankurtaran, Dalbastı Sk. No:14, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 638 90 10	https://www.bluehouseistanbul.com/	2	\N	\N
473	Viva Deluxe Hotel	\N	Eminönü, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
474	Hotel Vicenza	\N	Eminönü, Kemal Paşa, Fevziye Cd. No:3, 34134 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 520 17 01	http://www.hotelvicenzaistanbul.com/	2	\N	\N
475	Matbah Restaurant	\N	Eminönü, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
476	Hotel Zurich Istanbul	\N	Eminönü, Balabanağa, Vidinli Tevfikpaşa Cd No:14, 34134 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 512 23 50	http://www.hotelzurichistanbul.com/	2	\N	\N
477	Tria Hotel Istanbul	\N	Eminönü, Cankurtaran, Terbıyık Sk. No:7, 34122 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 45 18	http://www.triahotelistanbul.com/	2	\N	\N
478	Ramada By Wyndham Istanbul Pera	\N	Eminönü, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
479	Darkhill Hotel	\N	Eminönü, Mimar Kemalettin, Koca Ragıppaşa Cd No:9, 34130 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 638 89 00	http://www.darkhillhotel.com/	2	\N	\N
480	Yasmak Sultan Hotel	\N	Eminönü, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
481	Orient Express & Spa by Orka Hotels	\N	Fatih, Old City Sirkeci, Hoca Paşa, Hüdavendigar Cd. No:24, 34120 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 520 71 60	http://www.orientexpresshotel.com/	2	\N	\N
482	Georges Hotel Galata	\N	Fatih, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
483	Gülhanepark Hotel & Spa	\N	Fatih, Hocapaşa Mh, Nöbethane Cd. No:1, 34112 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 519 68 68	http://www.hotelgulhanepark.com/	2	\N	\N
484	Legacy Ottoman Hotel	\N	Fatih, Hobyar, Hamidiye Cd. No:16, 34112 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 527 67 67	http://www.legacyottomanhotel.com/	2	\N	\N
485	Orka Royal Hotel	\N	Fatih, Sirkeci Old City, Hoca Paşa, Nöbethane Cd. No:6, 34113 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 511 85 85	http://orkaroyalhotel.com/	2	\N	\N
486	Antik Hotel Istanbul	\N	Fatih, Mimar Kemalettin Mahallesi, Beyazıt, Sekbanbaşı Sk. No:6, 34130 Fatih/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 638 58 58	http://www.antikhotel.com/	2	\N	\N
487	Pierre Loti Hotel	\N	Fatih, Binbirdirek, Piyer Loti Cd. No:1, 34400 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 518 57 00	https://www.pierrelotihotel.com/	2	\N	\N
488	Viva Deluxe Hotel	\N	Fatih, Sirkeci, Hoca Paşa, İbni Kemal Cd. No:20, 34110 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 514 16 29	\N	2	\N	\N
489	Hotel Vicenza	\N	Fatih, Kemal Paşa, Fevziye Cd. No:3, 34134 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 520 17 01	http://www.hotelvicenzaistanbul.com/	2	\N	\N
490	Prestige Hotel Istanbul Laleli	\N	Fatih, Mimar Kemalettin, Koska Cd. No: 6, 34130 Fatih/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 518 82 80	http://www.hotelprestige.com.tr/	2	\N	\N
491	Ramada By Wyndham Istanbul Pera	\N	Fatih, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
492	Hotel Zurich Istanbul	\N	Fatih, Balabanağa, Vidinli Tevfikpaşa Cd No:14, 34134 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 512 23 50	http://www.hotelzurichistanbul.com/	2	\N	\N
493	Hotel Spectra	\N	Fatih, Binbirdirek, Şht. Mehmetpaşa Ykş. No:2, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 516 35 46	http://www.hotelspectra.com/	2	\N	\N
494	Matbah Restaurant	\N	Fatih, Cankurtaran, Caferiye Sk. No:6 D:1, 34122 Fatih/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 514 61 51	https://www.matbahrestaurant.com/	2	\N	\N
495	Yasmak Sultan Hotel	\N	Fatih, Hoca Paşa, Ebussuud Cd. No:12, 34110 Fatih/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 528 13 43	http://www.hotelyasmaksultan.com/	2	\N	\N
496	Istanbul Town Hotel	\N	Fatih, Mimar Kemalettin, Kalaycı Şevki Sk. No:4, 34126 Fatih/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 458 00 27	https://www.istanbultownhotel.com/	2	\N	\N
497	Darkhill Hotel	\N	Fatih, Mimar Kemalettin, Koca Ragıppaşa Cd No:9, 34130 Fatih/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 638 89 00	http://www.darkhillhotel.com/	2	\N	\N
498	Akgun Hotel Beyazit	\N	Fatih, Mimar Kemalettin, Haznedar Sk. No:6, 34490 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 638 28 38	http://www.akgunotel.com.tr/	2	\N	\N
499	Anemon Koleksiyon Galata Otel	\N	Fatih, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
500	Hotel Budo	\N	Fatih, Kemal Paşa, Laleli Cd. No:53, 34470 Fatih/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 526 27 27	http://www.budohotel.com/	2	\N	\N
501	Georges Hotel Galata	\N	Taksim, Müeyyetzade, Serdar-ı Ekrem Cd. No:24, 34425 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 244 24 23	http://www.georges.com/	2	\N	\N
502	Nova Plaza Orion Hotel	\N	Taksim, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
503	Swissôtel The Bosphorus Istanbul	\N	Taksim, Vişnezade, Acısu Sk. NO 19, 34357 Beşiktaş/İstanbul, Türkiye	4.8	local_database	\N	\N	(0212) 326 11 00	https://www.swissotel.com/hotels/istanbul/?goto=fiche_hotel&code_hotel=A5D2&merchantid=seo-maps-TR-A5D2&sourceid=aw-cen&utm_medium=seo%20maps&utm_source=google%20Maps&utm_campaign=seo%20maps	2	\N	\N
504	TAKSİM STAR HOTEL BOSPHORUS	\N	Taksim, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
505	Ramada By Wyndham Istanbul Pera	\N	Taksim, Asmalı Mescit, Oteller Sk. No:1 D:3, 34430 Beyoğlu/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 252 71 60	https://www.wyndhamhotels.com/ramada/istanbul-turkiye/ramada-istanbul-pera/overview?CID=LC:wmcic5n98gs1g0r:58105&iata=00093796	2	\N	\N
506	CVK Park Bosphorus Hotel Istanbul	\N	Taksim, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
507	Avantgarde Urban Hotel Taksim	\N	Taksim, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
508	dora hotel	\N	Taksim, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
509	Ferman Hilal Hotel	\N	Taksim, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
510	Anemon Koleksiyon Galata Otel	\N	Taksim, Mahallesi, Bereketzade, Büyük Hendek Cd. No:5, 34421 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 919 09 70	https://anemonhotels.com/portfolio/anemon-galata/	2	\N	\N
511	360 Istanbul	\N	Taksim, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
512	Orka Taksim Suites & Hotel	\N	Taksim, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
513	Nippon Hotel	\N	Taksim, Kocatepe, Topçu Cd. No:6, 34437 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 313 33 00	http://www.nipponhotel.com.tr/	2	\N	\N
514	Opera Hotel Bosphorus	\N	Taksim, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
515	BellaVista Hostel	\N	Taksim, Bülbül, Turan Cd. No:50, 34400 Beyoğlu/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 361 92 55	\N	2	\N	\N
516	Konak Hotel	\N	Taksim, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
517	Buyuk Londra Hotel's Terrace Bar	\N	Taksim, Asmalı Mescit, Meşrutiyet Cd. No:53, 34430 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 245 06 70	https://londrahotel.net/tr/	2	\N	\N
518	Mr CAS Hotels	\N	Taksim, Kuloğlu, İstiklal Cd. No:153, 34433 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 00 07	https://www.mrcashotels.com/	2	\N	\N
519	Van Kahvaltı Evi	\N	Taksim, Kılıçali Paşa, Defterdar Ykş. 52/A, 34425 Beyoğlu/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 293 64 37	\N	2	\N	\N
520	5. Kat Restaurant	\N	Taksim, Cihangir, Soğancı Sk. No:3, 34427, 34433 Beyoğlu/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 293 37 74	http://www.5kat.com/	2	\N	\N
521	The Cettia Istanbul	\N	Şişli, Fulya Mahallesi, Mecidiyeköy, Ortaklar Cd. No:30, 34834 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 912 43 62	https://www.thecettiaistanbul.com/	2	\N	\N
522	Holiday Inn Istanbul Sisli	\N	Şişli, 19 Mayıs, Halaskargazi Cd. No:206, 34360 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 373 38 00	http://www.hisisli.com/	2	\N	\N
523	dora hotel	\N	Şişli, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
524	Burger King	\N	Şişli, Fulya, Prof. Dr. Bülent Tarcan Sk. No:86, 34394 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
525	Konak Hotel	\N	Şişli, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
526	Domino's Pizza	\N	Şişli, Gayrettepe, Ortaklar Cd. 42/A, 34394 Şişli/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 347 33 99	https://www.dominos.com.tr/subeler/istanbul/gayrettepe-42804	2	\N	\N
527	Domino's Pizza Şişli	\N	Şişli, Akın, Cumhuriyet, Silahşör Cd. No:39 Apt. 37, 34380 Şişli/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 231 48 00	https://www.dominos.com.tr/subeler/istanbul/sisli-42821	2	\N	\N
528	Nişantaşı Başköşe	\N	Şişli, Harbiye, Bronz Sk. No:5/1, 34370 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 230 38 68	http://www.nisantasibaskose.com/	2	\N	\N
529	Burger King	\N	Şişli, 19 Mayıs, Büyükdere Cd. No:24, 34360 Şişli/İstanbul, Türkiye	3.3	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
530	Little Caesars Fulya Şubesi	\N	Şişli, 19 Mayıs Caddesi, 19 Mayıs, Aşçılar Sk. No:3/B, 34394 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 274 35 55	https://www.littlecaesars.com.tr/?utm_source=GMB&utm_medium=local&utm_campaign=LC-Ortaklar-Profile-Click	2	\N	\N
531	Cafe Zone	\N	Şişli, Halaskargazi, Kuyumcu İrfan Sk. 3/B, 34371 Şişli/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 296 65 91	https://www.cafezone.com.tr/	2	\N	\N
532	Garden Iskender	\N	Şişli, Meşrutiyet Mh. Halaskargazi Cd. Ebe Kızı Sk., No:2 Osmanbey, Meşrutiyet, 34363 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 247 45 77	http://gardeniskender.com/	2	\N	\N
533	Dragon Restaurant	\N	Şişli, Harbiye, Cumhuriyet Cd. No:50, 34367 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 231 62 00	http://dragonrestaurant.com.tr/	2	\N	\N
534	Zorba	\N	Şişli, Gayrettepe, Yıldız Posta Cd. No:28 D:g, 34349 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 267 13 12	http://www.zorbataverna.com.tr/	2	\N	\N
535	Rumeli Köftecisi	\N	Şişli, Fulya, Bahçeler Sokağı No:9, 34394 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 274 38 84	\N	2	\N	\N
536	Özlem Kebap	\N	Şişli, Kurtuluş Mh, Kurtuluş Cd. 20/A, 34379 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 231 00 62	\N	2	\N	\N
537	Pizzeria 14	\N	Şişli, Teşvikiye Mahallesi Güzel Bahçe Sok., Nişantaşı Hancıoğlu Apt, Teşvikiye, D:14/A, 34365 Şişli/İstanbul, Türkiye	4.6	local_database	\N	\N	(0212) 343 86 60	\N	2	\N	\N
538	Bolpi Pide ve Lahmacun	\N	Şişli, Fulya Mahallesi, Mecidiyeköy, Mehmetçik Cd. 4/A, 34394 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 211 19 00	\N	2	\N	\N
539	Izzetpasa Kokorec	\N	Şişli, İzzet Paşa, Yeni Yol Cd. No:26, 34387 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 224 87 49	\N	2	\N	\N
540	Sarıhan İşkembe Şişli	\N	Şişli, Merkez, Abide-i Hürriyet Cd No: 126, 34400 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 233 93 21	\N	2	\N	\N
541	Nova Plaza Orion Hotel	\N	Nişantaşı, Kocatepe, Lamartin Cd. No:22, 34437 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 236 66 82	https://www.novaplazahotels.com/	2	\N	\N
542	Swissôtel The Bosphorus Istanbul	\N	Nişantaşı, Vişnezade, Acısu Sk. NO 19, 34357 Beşiktaş/İstanbul, Türkiye	4.8	local_database	\N	\N	(0212) 326 11 00	https://www.swissotel.com/hotels/istanbul/?goto=fiche_hotel&code_hotel=A5D2&merchantid=seo-maps-TR-A5D2&sourceid=aw-cen&utm_medium=seo%20maps&utm_source=google%20Maps&utm_campaign=seo%20maps	2	\N	\N
543	TAKSİM STAR HOTEL BOSPHORUS	\N	Nişantaşı, Cihangir, Sıraselviler Cd. No:37, 34433 Beyoğlu/İstanbul, Türkiye	2.9	local_database	\N	\N	(0212) 293 80 80	http://www.taksimstarhotel.com/	2	\N	\N
544	CVK Park Bosphorus Hotel Istanbul	\N	Nişantaşı, Gümüşsuyu, İnönü Cd. No: 8, 34437 Beyoğlu/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 377 88 88	http://www.cvkhotelsandresorts.com/tr/	2	\N	\N
545	Avantgarde Urban Hotel Taksim	\N	Nişantaşı, Kocatepe, Abdülhak Hamit Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 235 81 60	https://www.avantgardecollection.com/urban-taksim-hotel	2	\N	\N
546	dora hotel	\N	Nişantaşı, Eskişehir, Dolapdere Cd. No:33, 34375 Şişli/İstanbul, Türkiye	3.2	local_database	\N	\N	(0212) 233 70 70	http://www.istanbuldora.com/	2	\N	\N
547	Holiday Inn Istanbul Sisli	\N	Nişantaşı, 19 Mayıs, Halaskargazi Cd. No:206, 34360 Şişli/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 373 38 00	http://www.hisisli.com/	2	\N	\N
548	Ferman Hilal Hotel	\N	Nişantaşı, Kocatepe, Topçu Cd. No:23, 34280 Beyoğlu/İstanbul, Türkiye	3.4	local_database	\N	\N	(0212) 256 13 80	http://www.fermanhilal.com/	2	\N	\N
549	The Cettia Istanbul	\N	Nişantaşı, Fulya Mahallesi, Mecidiyeköy, Ortaklar Cd. No:30, 34834 Şişli/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 912 43 62	https://www.thecettiaistanbul.com/	2	\N	\N
550	Nippon Hotel	\N	Nişantaşı, Kocatepe, Topçu Cd. No:6, 34437 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 313 33 00	http://www.nipponhotel.com.tr/	2	\N	\N
551	Opera Hotel Bosphorus	\N	Nişantaşı, Ömer Avni, Gümüşsuyu, İnönü Cd. No:26, 34427 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 372 45 00	https://www.operahotel.com.tr/	2	\N	\N
552	BellaVista Hostel	\N	Nişantaşı, Bülbül, Turan Cd. No:50, 34400 Beyoğlu/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 361 92 55	\N	2	\N	\N
553	Konak Hotel	\N	Nişantaşı, İnönü, Cumhuriyet Cd. No:75, 34373 Şişli/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 225 82 50	http://www.konakhotel.com/	2	\N	\N
554	Orka Taksim Suites & Hotel	\N	Nişantaşı, Kuloğlu Mh, Cihangir, Ağa Hamamı Sk. No:18, 34433 Beyoğlu/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 243 40 66	http://www.orkataksim.com/	2	\N	\N
555	360 Istanbul	\N	Nişantaşı, Tomtom Mah. İstiklal Cad. No:163 K: 8, Tomtom, 34433 Beyoğlu/İstanbul, Türkiye	4	local_database	\N	\N	0533 691 03 60	http://www.360istanbul.com/	2	\N	\N
556	5. Kat Restaurant	\N	Nişantaşı, Cihangir, Soğancı Sk. No:3, 34427, 34433 Beyoğlu/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 293 37 74	http://www.5kat.com/	2	\N	\N
557	Mr CAS Hotels	\N	Nişantaşı, Kuloğlu, İstiklal Cd. No:153, 34433 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 293 00 07	https://www.mrcashotels.com/	2	\N	\N
558	More Restaurant	\N	Nişantaşı, Kocatepe, Lamartin Cd. No:13, 34437 Beyoğlu/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 361 67 67	https://www.midtown-hotel.com/tr/hotel/more-restaurant/	2	\N	\N
559	Topaz İstanbul	\N	Nişantaşı, Ömer Avni, İnönü Cd. No:42, 34437 Beyoğlu/İstanbul, Türkiye	4.2	local_database	\N	\N	0531 329 41 11	http://www.topazistanbul.com/	2	\N	\N
560	Tarihi Cumhuriyet Meyhanesi	\N	Nişantaşı, Kamer Hatun Sahne Sokak, Hüseyinağa, Kamer Hatun Cd. No:27, 34435 Beyoğlu/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 252 08 86	http://www.tarihicumhuriyetmeyhanesi.com.tr/	2	\N	\N
561	Avantgarde Urban Hotel Levent	\N	Etiler, Esentepe, Büyükdere Cd. No:161, 34394 Şişli/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 337 04 44	https://www.avantgardecollection.com/urban-levent-hotel	2	\N	\N
562	Sunset Grill & Bar	\N	Etiler, Kuruçeşme Mahallesi Ulus Park, Kuruçeşme, Yol Sokağı No:2, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 287 03 57	https://www.sunsetgrillbar.com/	2	\N	\N
563	Mangerie	\N	Etiler, Bebek, Cevdet Paşa Cd. No:69, 34342 Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 263 51 99	http://www.mangeriebebek.com/	2	\N	\N
564	The House Cafe	\N	Etiler, Levent, Kanyon AVM No:185, 34394 Beşiktaş/İstanbul, Türkiye	3.3	local_database	\N	\N	(0212) 353 53 75	\N	2	\N	\N
565	Vitrin Meyhane Etiler	\N	Etiler, Nispetiye Mahallesi, Aytar Cd. 3/A, 34443 Etiler/Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 232 20 46	\N	2	\N	\N
566	Develi Etiler	\N	Etiler, Etiler, Tepecik Yolu No:22 D:22, 34337 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 263 25 71	http://www.develikebap.com/	2	\N	\N
567	Paper Moon	\N	Etiler, Etiler, Ahmet Adnan Saygun Cd., 34340 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 282 16 16	https://www.papermoonrestaurants.com/paper-moon-istanbul.html	2	\N	\N
568	Midpoint Bebek	\N	Etiler, Bebek, Cevdet Paşa Cd. No: 35, 34342 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	0530 918 00 26	https://www.midpoint.com.tr/	2	\N	\N
569	Ulus Cafe	\N	Etiler, Kuruçeşme Adnan Saygun Cad, Kuruçeşme, Ulus Parkı 71/B, 34345 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 263 84 37	http://uluscafe.com/	2	\N	\N
570	Köfteci Ramiz	\N	Etiler, Levent, Büyükdere Cd. No:224, 34330 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 344 04 31	https://www.google.com/url?sa=D&q=http://www.kofteciramiz.com/	2	\N	\N
571	Sevgi simit (Yücel Simit)	\N	Etiler, Nisbetiye, Nisbetiye Cd No:32, 34340 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 270 21 51	\N	2	\N	\N
572	Karafırın Ulus	\N	Etiler, Kültür, Ahmet Adnan Saygun Cd. 39/A, 34340 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 265 21 51	https://www.karafirin.com.tr/	2	\N	\N
573	Bolulu Hasan Usta Süt Tatlıları	\N	Etiler, Nisbetiye, Nisbetiye Cd No:28/11, 34340 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 264 16 64	http://www.bhu.com.tr/	2	\N	\N
574	Galatasaraylılar Derneği Lokali	\N	Etiler, Levent, 34335 Beşiktaş/İstanbul, Türkiye	4.5	local_database	\N	\N	0538 812 18 68	https://galatasaraylilardernegi.org.tr/	2	\N	\N
575	Naturel Park Spor Tesisleri	\N	Etiler, Akatlar Mahallesi, Etiler, Ebulula Mardin Cd. No:2/1, 34335 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 352 70 65	https://www.instagram.com/etilernaturelpark	2	\N	\N
576	Gunaydin	\N	Etiler, Etiler, Nisbetiye Cd No:104 D:F, 34337 Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 282 01 53	https://www.gunaydinkoftedoner.com/	2	\N	\N
577	Diet Club	\N	Etiler, Levent, Hanımeli Sk. No:5, 34330 Beşiktaş/İstanbul, Türkiye	1.9	local_database	\N	\N	(0212) 270 66 11	\N	2	\N	\N
578	BURGERLAB	\N	Etiler, Food Court, Kültür, Ak Merkez AVM, 34337 Beşiktaş/İstanbul, Türkiye	3.9	local_database	\N	\N	(0212) 282 18 18	\N	2	\N	\N
579	McDonald's İstanbul Özdilek Avm	\N	Etiler, Esentepe Büyükdere Caddesi Özdilek AVM, Esentepe, D:177/43, 34394 Şişli/İstanbul, Türkiye	2.7	local_database	\N	\N	0533 281 00 30	https://www.mcdonalds.com.tr/kurumsal/restoranlar	2	\N	\N
580	Ulubey Manti	\N	Etiler, 120 A, Etiler, Nisbetiye Cd, 34337 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 265 01 57	\N	2	\N	\N
581	Avantgarde Urban Hotel Levent	\N	Levent, Esentepe, Büyükdere Cd. No:161, 34394 Şişli/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 337 04 44	https://www.avantgardecollection.com/urban-levent-hotel	2	\N	\N
582	The House Cafe	\N	Levent, Levent, Kanyon AVM No:185, 34394 Beşiktaş/İstanbul, Türkiye	3.3	local_database	\N	\N	(0212) 353 53 75	\N	2	\N	\N
583	Vitrin Meyhane Etiler	\N	Levent, Nispetiye Mahallesi, Aytar Cd. 3/A, 34443 Etiler/Beşiktaş/İstanbul, Türkiye	3.5	local_database	\N	\N	(0212) 232 20 46	\N	2	\N	\N
584	Paper Moon	\N	Levent, Etiler, Ahmet Adnan Saygun Cd., 34340 Beşiktaş/İstanbul, Türkiye	4.4	local_database	\N	\N	(0212) 282 16 16	https://www.papermoonrestaurants.com/paper-moon-istanbul.html	2	\N	\N
585	Develi Etiler	\N	Levent, Etiler, Tepecik Yolu No:22 D:22, 34337 Beşiktaş/İstanbul, Türkiye	4.2	local_database	\N	\N	(0212) 263 25 71	http://www.develikebap.com/	2	\N	\N
586	Burger King	\N	Levent, Fulya, Prof. Dr. Bülent Tarcan Sk. No:86, 34394 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	444 5 464	https://www.burgerking.com.tr/	2	\N	\N
587	Köfteci Ramiz	\N	Levent, Levent, Büyükdere Cd. No:224, 34330 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 344 04 31	https://www.google.com/url?sa=D&q=http://www.kofteciramiz.com/	2	\N	\N
588	Kayra Life Restaurant	\N	Levent, Konaklar, Akasyalı Sk. No:3 No:3, 34330 Beşiktaş/İstanbul, Türkiye	3.8	local_database	\N	\N	(0212) 279 27 10	\N	2	\N	\N
589	Sevgi simit (Yücel Simit)	\N	Levent, Nisbetiye, Nisbetiye Cd No:32, 34340 Beşiktaş/İstanbul, Türkiye	3.6	local_database	\N	\N	(0212) 270 21 51	\N	2	\N	\N
590	Galatasaraylılar Derneği Lokali	\N	Levent, Levent, 34335 Beşiktaş/İstanbul, Türkiye	4.5	local_database	\N	\N	0538 812 18 68	https://galatasaraylilardernegi.org.tr/	2	\N	\N
591	Bolulu Hasan Usta Süt Tatlıları	\N	Levent, Nisbetiye, Nisbetiye Cd No:28/11, 34340 Beşiktaş/İstanbul, Türkiye	4.1	local_database	\N	\N	(0212) 264 16 64	http://www.bhu.com.tr/	2	\N	\N
592	Naturel Park Spor Tesisleri	\N	Levent, Akatlar Mahallesi, Etiler, Ebulula Mardin Cd. No:2/1, 34335 Beşiktaş/İstanbul, Türkiye	4.3	local_database	\N	\N	(0212) 352 70 65	https://www.instagram.com/etilernaturelpark	2	\N	\N
593	McDonald's İstanbul Özdilek Avm	\N	Levent, Esentepe Büyükdere Caddesi Özdilek AVM, Esentepe, D:177/43, 34394 Şişli/İstanbul, Türkiye	2.7	local_database	\N	\N	0533 281 00 30	https://www.mcdonalds.com.tr/kurumsal/restoranlar	2	\N	\N
594	Hışım Lezzet Lokantası	\N	Levent, Sultan Selim, Turan Sokağı No:4, 34415 Kağıthane/İstanbul, Türkiye	4.5	local_database	\N	\N	(0212) 281 72 32	http://www.hisimborek.com.tr/	2	\N	\N
595	Diet Club	\N	Levent, Levent, Hanımeli Sk. No:5, 34330 Beşiktaş/İstanbul, Türkiye	1.9	local_database	\N	\N	(0212) 270 66 11	\N	2	\N	\N
596	Yesillik	\N	Levent, Levent, Büyükdere Cd. Metrocity Alışveriş Merkezi No:171 D:252, 34394 Beşiktaş/Istanbul - Europe, Türkiye	\N	local_database	\N	\N	(0212) 344 00 04	\N	2	\N	\N
597	Karanlıkta Yemek	\N	Levent, Sultan Selim, Akyol Sitesi Çk No:10, 34415 Kağıthane/İstanbul, Türkiye	4.5	local_database	\N	\N	0532 342 25 38	http://www.karanliktayemek.com/	2	\N	\N
598	Pilove	\N	Levent, Esentepe, Eski Büyükdere Cd. 171/237, 34330 Şişli/İstanbul, Türkiye	3.7	local_database	\N	\N	(0212) 344 06 00	\N	2	\N	\N
599	Konak	\N	Levent, Levent, Büyükdere Cd. No:2, 34330 Beşiktaş/İstanbul, Türkiye	2.4	local_database	\N	\N	(0212) 280 85 61	\N	2	\N	\N
600	Le Pain Quotidien	\N	Levent, Esentepe, Kanyon AVM No:185, 34330 Şişli/İstanbul, Türkiye	4	local_database	\N	\N	(0212) 450 50 20	https://lepainquotidien.com/tr/stores/kanyon%20mall?utm_source=gmb&utm_medium=business-listing	2	\N	\N
\.


--
-- TOC entry 4489 (class 0 OID 16610)
-- Dependencies: 251
-- Data for Name: route_history; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.route_history (id, user_id, navigation_session_id, origin, destination, waypoints, distance, duration, transport_mode, route_geometry, steps, user_rating, user_feedback, completed_at) FROM stdin;
\.


--
-- TOC entry 4490 (class 0 OID 16618)
-- Dependencies: 252
-- Data for Name: user_interaction_aggregates; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.user_interaction_aggregates (id, user_id, item_type, view_count, click_count, save_count, rating_count, conversion_count, avg_rating, avg_dwell_time, click_through_rate, conversion_rate, last_interaction, recency_score, category_preferences, updated_at) FROM stdin;
\.


--
-- TOC entry 4491 (class 0 OID 16626)
-- Dependencies: 253
-- Data for Name: user_preferences; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.user_preferences (id, user_id, preferred_transport, avoid_highways, avoid_tolls, avoid_ferries, wheelchair_accessible, requires_elevator, preferred_language, distance_units, voice_guidance, notification_sound, vibration, save_location_history, share_location, interests, dietary_restrictions, budget_level, created_at, updated_at) FROM stdin;
\.


--
-- TOC entry 4492 (class 0 OID 16633)
-- Dependencies: 254
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.users (id, name, email) FROM stdin;
\.


--
-- TOC entry 4498 (class 0 OID 0)
-- Dependencies: 223
-- Name: blog_posts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.blog_posts_id_seq', 23, true);


--
-- TOC entry 4499 (class 0 OID 0)
-- Dependencies: 222
-- Name: chat_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.chat_history_id_seq', 1, true);


--
-- TOC entry 4500 (class 0 OID 0)
-- Dependencies: 233
-- Name: chat_sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.chat_sessions_id_seq', 1, true);


--
-- TOC entry 4501 (class 0 OID 0)
-- Dependencies: 234
-- Name: conversation_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.conversation_history_id_seq', 1, true);


--
-- TOC entry 4502 (class 0 OID 0)
-- Dependencies: 221
-- Name: events_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.events_id_seq', 13, true);


--
-- TOC entry 4503 (class 0 OID 0)
-- Dependencies: 224
-- Name: feedback_events_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.feedback_events_id_seq', 1, true);


--
-- TOC entry 4504 (class 0 OID 0)
-- Dependencies: 228
-- Name: intent_feedback_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.intent_feedback_id_seq', 1, true);


--
-- TOC entry 4505 (class 0 OID 0)
-- Dependencies: 226
-- Name: item_feature_vectors_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.item_feature_vectors_id_seq', 1, true);


--
-- TOC entry 4506 (class 0 OID 0)
-- Dependencies: 229
-- Name: location_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.location_history_id_seq', 1, true);


--
-- TOC entry 4507 (class 0 OID 0)
-- Dependencies: 219
-- Name: museums_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.museums_id_seq', 60, true);


--
-- TOC entry 4508 (class 0 OID 0)
-- Dependencies: 231
-- Name: navigation_events_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.navigation_events_id_seq', 1, true);


--
-- TOC entry 4509 (class 0 OID 0)
-- Dependencies: 230
-- Name: navigation_sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.navigation_sessions_id_seq', 1, true);


--
-- TOC entry 4510 (class 0 OID 0)
-- Dependencies: 227
-- Name: online_learning_models_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.online_learning_models_id_seq', 1, true);


--
-- TOC entry 4511 (class 0 OID 0)
-- Dependencies: 218
-- Name: places_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.places_id_seq', 1, true);


--
-- TOC entry 4512 (class 0 OID 0)
-- Dependencies: 220
-- Name: restaurants_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.restaurants_id_seq', 600, true);


--
-- TOC entry 4513 (class 0 OID 0)
-- Dependencies: 235
-- Name: route_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.route_history_id_seq', 1, true);


--
-- TOC entry 4514 (class 0 OID 0)
-- Dependencies: 225
-- Name: user_interaction_aggregates_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.user_interaction_aggregates_id_seq', 1, true);


--
-- TOC entry 4515 (class 0 OID 0)
-- Dependencies: 232
-- Name: user_preferences_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.user_preferences_id_seq', 1, true);


--
-- TOC entry 4516 (class 0 OID 0)
-- Dependencies: 217
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.users_id_seq', 1, true);


--
-- TOC entry 4263 (class 1259 OID 16512)
-- Name: events_biletix_id_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX events_biletix_id_key ON public.events USING btree (biletix_id);


--
-- TOC entry 4272 (class 1259 OID 16545)
-- Name: idx_feedback_quality; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_feedback_quality ON public.intent_feedback USING btree (feedback_type, is_correct, "timestamp");


--
-- TOC entry 4273 (class 1259 OID 16542)
-- Name: idx_feedback_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_feedback_status ON public.intent_feedback USING btree (review_status, used_for_training);


--
-- TOC entry 4274 (class 1259 OID 16534)
-- Name: idx_training_data; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_training_data ON public.intent_feedback USING btree (used_for_training, review_status, predicted_intent);


--
-- TOC entry 4256 (class 1259 OID 16487)
-- Name: ix_chat_history_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_history_session_id ON public.chat_history USING btree (session_id);


--
-- TOC entry 4257 (class 1259 OID 16496)
-- Name: ix_chat_sessions_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_chat_sessions_session_id ON public.chat_sessions USING btree (session_id);


--
-- TOC entry 4258 (class 1259 OID 16494)
-- Name: ix_chat_sessions_started_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_sessions_started_at ON public.chat_sessions USING btree (started_at);


--
-- TOC entry 4259 (class 1259 OID 16495)
-- Name: ix_chat_sessions_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_sessions_user_id ON public.chat_sessions USING btree (user_id);


--
-- TOC entry 4260 (class 1259 OID 16504)
-- Name: ix_conversation_history_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_conversation_history_session_id ON public.conversation_history USING btree (session_id);


--
-- TOC entry 4261 (class 1259 OID 16505)
-- Name: ix_conversation_history_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_conversation_history_timestamp ON public.conversation_history USING btree ("timestamp");


--
-- TOC entry 4262 (class 1259 OID 16503)
-- Name: ix_conversation_history_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_conversation_history_user_id ON public.conversation_history USING btree (user_id);


--
-- TOC entry 4264 (class 1259 OID 16514)
-- Name: ix_events_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_events_id ON public.events USING btree (id);


--
-- TOC entry 4265 (class 1259 OID 16513)
-- Name: ix_events_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_events_name ON public.events USING btree (name);


--
-- TOC entry 4266 (class 1259 OID 16525)
-- Name: ix_feedback_events_event_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feedback_events_event_type ON public.feedback_events USING btree (event_type);


--
-- TOC entry 4267 (class 1259 OID 16523)
-- Name: ix_feedback_events_item_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feedback_events_item_id ON public.feedback_events USING btree (item_id);


--
-- TOC entry 4268 (class 1259 OID 16521)
-- Name: ix_feedback_events_processed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feedback_events_processed ON public.feedback_events USING btree (processed);


--
-- TOC entry 4269 (class 1259 OID 16522)
-- Name: ix_feedback_events_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feedback_events_session_id ON public.feedback_events USING btree (session_id);


--
-- TOC entry 4270 (class 1259 OID 16524)
-- Name: ix_feedback_events_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feedback_events_timestamp ON public.feedback_events USING btree ("timestamp");


--
-- TOC entry 4271 (class 1259 OID 16526)
-- Name: ix_feedback_events_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feedback_events_user_id ON public.feedback_events USING btree (user_id);


--
-- TOC entry 4275 (class 1259 OID 16544)
-- Name: ix_intent_feedback_actual_intent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_actual_intent ON public.intent_feedback USING btree (actual_intent);


--
-- TOC entry 4276 (class 1259 OID 16541)
-- Name: ix_intent_feedback_feedback_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_feedback_type ON public.intent_feedback USING btree (feedback_type);


--
-- TOC entry 4277 (class 1259 OID 16543)
-- Name: ix_intent_feedback_is_correct; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_is_correct ON public.intent_feedback USING btree (is_correct);


--
-- TOC entry 4278 (class 1259 OID 16540)
-- Name: ix_intent_feedback_language; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_language ON public.intent_feedback USING btree (language);


--
-- TOC entry 4279 (class 1259 OID 16536)
-- Name: ix_intent_feedback_predicted_intent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_predicted_intent ON public.intent_feedback USING btree (predicted_intent);


--
-- TOC entry 4280 (class 1259 OID 16535)
-- Name: ix_intent_feedback_review_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_review_status ON public.intent_feedback USING btree (review_status);


--
-- TOC entry 4281 (class 1259 OID 16537)
-- Name: ix_intent_feedback_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_session_id ON public.intent_feedback USING btree (session_id);


--
-- TOC entry 4282 (class 1259 OID 16538)
-- Name: ix_intent_feedback_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_timestamp ON public.intent_feedback USING btree ("timestamp");


--
-- TOC entry 4283 (class 1259 OID 16539)
-- Name: ix_intent_feedback_used_for_training; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_used_for_training ON public.intent_feedback USING btree (used_for_training);


--
-- TOC entry 4284 (class 1259 OID 16533)
-- Name: ix_intent_feedback_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_intent_feedback_user_id ON public.intent_feedback USING btree (user_id);


--
-- TOC entry 4285 (class 1259 OID 16553)
-- Name: ix_item_feature_vectors_item_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_item_feature_vectors_item_id ON public.item_feature_vectors USING btree (item_id);


--
-- TOC entry 4287 (class 1259 OID 16560)
-- Name: ix_location_history_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_location_history_session_id ON public.location_history USING btree (session_id);


--
-- TOC entry 4288 (class 1259 OID 16561)
-- Name: ix_location_history_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_location_history_timestamp ON public.location_history USING btree ("timestamp");


--
-- TOC entry 4289 (class 1259 OID 16559)
-- Name: ix_location_history_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_location_history_user_id ON public.location_history USING btree (user_id);


--
-- TOC entry 4290 (class 1259 OID 16568)
-- Name: ix_museums_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_museums_id ON public.museums USING btree (id);


--
-- TOC entry 4291 (class 1259 OID 16569)
-- Name: ix_museums_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_museums_name ON public.museums USING btree (name);


--
-- TOC entry 4292 (class 1259 OID 16578)
-- Name: ix_navigation_events_event_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_events_event_type ON public.navigation_events USING btree (event_type);


--
-- TOC entry 4293 (class 1259 OID 16577)
-- Name: ix_navigation_events_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_events_session_id ON public.navigation_events USING btree (session_id);


--
-- TOC entry 4294 (class 1259 OID 16579)
-- Name: ix_navigation_events_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_events_timestamp ON public.navigation_events USING btree ("timestamp");


--
-- TOC entry 4295 (class 1259 OID 16576)
-- Name: ix_navigation_events_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_events_user_id ON public.navigation_events USING btree (user_id);


--
-- TOC entry 4296 (class 1259 OID 16589)
-- Name: ix_navigation_sessions_chat_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_sessions_chat_session_id ON public.navigation_sessions USING btree (chat_session_id);


--
-- TOC entry 4297 (class 1259 OID 16587)
-- Name: ix_navigation_sessions_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_navigation_sessions_session_id ON public.navigation_sessions USING btree (session_id);


--
-- TOC entry 4298 (class 1259 OID 16586)
-- Name: ix_navigation_sessions_started_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_sessions_started_at ON public.navigation_sessions USING btree (started_at);


--
-- TOC entry 4299 (class 1259 OID 16588)
-- Name: ix_navigation_sessions_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_navigation_sessions_user_id ON public.navigation_sessions USING btree (user_id);


--
-- TOC entry 4301 (class 1259 OID 16609)
-- Name: ix_restaurants_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_restaurants_id ON public.restaurants USING btree (id);


--
-- TOC entry 4302 (class 1259 OID 16608)
-- Name: ix_restaurants_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_restaurants_name ON public.restaurants USING btree (name);


--
-- TOC entry 4304 (class 1259 OID 16617)
-- Name: ix_route_history_completed_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_route_history_completed_at ON public.route_history USING btree (completed_at);


--
-- TOC entry 4305 (class 1259 OID 16616)
-- Name: ix_route_history_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_route_history_user_id ON public.route_history USING btree (user_id);


--
-- TOC entry 4306 (class 1259 OID 16625)
-- Name: ix_user_interaction_aggregates_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_user_interaction_aggregates_user_id ON public.user_interaction_aggregates USING btree (user_id);


--
-- TOC entry 4308 (class 1259 OID 16632)
-- Name: ix_user_preferences_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_user_preferences_user_id ON public.user_preferences USING btree (user_id);


--
-- TOC entry 4300 (class 1259 OID 16596)
-- Name: online_learning_models_model_name_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX online_learning_models_model_name_key ON public.online_learning_models USING btree (model_name);


--
-- TOC entry 4303 (class 1259 OID 16607)
-- Name: restaurants_place_id_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX restaurants_place_id_key ON public.restaurants USING btree (place_id);


--
-- TOC entry 4286 (class 1259 OID 16552)
-- Name: uix_item_id_type; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uix_item_id_type ON public.item_feature_vectors USING btree (item_id, item_type);


--
-- TOC entry 4307 (class 1259 OID 16624)
-- Name: uix_user_item_type; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uix_user_item_type ON public.user_interaction_aggregates USING btree (user_id, item_type);


--
-- TOC entry 4309 (class 1259 OID 16637)
-- Name: users_email_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX users_email_key ON public.users USING btree (email);


-- Completed on 2025-12-11 15:24:18 +03

--
-- PostgreSQL database dump complete
--

\unrestrict hv3IH62ThRFDsBaEgxogkwBPAxTcLYKVNvBtz4kPhCqx0uxoQUBo0uWAyy40DAq

