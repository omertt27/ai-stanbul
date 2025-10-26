#!/usr/bin/env python3
"""
İKSV Selenium Event Scraper
Uses Selenium WebDriver to scrape JavaScript-rendered event data from İKSV websites
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

class IKSVSeleniumScraper:
    """Selenium-based scraper for İKSV event websites"""
    
    def __init__(self, headless: bool = True):
        """
        Initialize Selenium scraper
        
        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.headless = headless
        self.driver = None
        
    def __enter__(self):
        """Context manager entry"""
        self._initialize_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup driver"""
        self.close()
    
    def _initialize_driver(self):
        """Initialize Chrome WebDriver with appropriate options"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument('--headless')
            
            # Additional options for stability and performance
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            # User agent to avoid detection
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Initialize driver
            logger.info("🚀 Initializing Chrome WebDriver...")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            
            logger.info("✅ WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebDriver: {e}")
            raise
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🔒 WebDriver closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing WebDriver: {e}")
    
    def scrape_main_calendar(self, url: str = "https://www.iksv.org/en") -> List[Dict[str, Any]]:
        """
        Scrape events from the main İKSV calendar section
        
        Args:
            url: URL to scrape (defaults to main İKSV site)
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        try:
            logger.info(f"🌐 Loading {url}...")
            self.driver.get(url)
            
            # Navigate to calendar section if not already there
            if '#event-calendar-section' not in url:
                try:
                    calendar_link = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="event-calendar-section"]'))
                    )
                    calendar_link.click()
                    logger.info("📅 Navigated to calendar section")
                except TimeoutException:
                    logger.warning("⚠️ Could not find calendar section link")
            
            # Wait for calendar section to load
            time.sleep(3)  # Give JavaScript time to render
            
            # Scroll to load lazy-loaded content
            self._scroll_page()
            
            # Extract events from calendar
            events = self._extract_calendar_events()
            
            logger.info(f"✅ Found {len(events)} events from main calendar")
            
        except Exception as e:
            logger.error(f"❌ Error scraping main calendar: {e}")
        
        return events
    
    def _scroll_page(self):
        """Scroll page to trigger lazy loading of content"""
        try:
            # Scroll down in steps to trigger lazy loading
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            current_position = 0
            scroll_step = 500
            
            while current_position < total_height:
                self.driver.execute_script(f"window.scrollTo(0, {current_position});")
                time.sleep(0.5)
                current_position += scroll_step
                # Update total height as new content loads
                total_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
        except Exception as e:
            logger.debug(f"Error during scrolling: {e}")
    
    def _extract_calendar_events(self) -> List[Dict[str, Any]]:
        """Extract events from the loaded calendar page"""
        events = []
        
        try:
            # Wait for event calendar list to be present
            try:
                calendar_list = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'event-calendar-list'))
                )
            except TimeoutException:
                logger.warning("⚠️ Event calendar list not found")
                return events
            
            # Try multiple selectors for event items
            event_selectors = [
                '.event-calendar-item',
                '.calendar-item',
                '.event-item',
                '[class*="event"]',
                '.card',
            ]
            
            event_elements = []
            for selector in event_selectors:
                try:
                    elements = calendar_list.find_elements(By.CSS_SELECTOR, selector)
                    if len(elements) > 0:
                        event_elements = elements
                        logger.info(f"📋 Found {len(elements)} elements with selector: {selector}")
                        break
                except NoSuchElementException:
                    continue
            
            if not event_elements:
                logger.warning("⚠️ No event elements found with any selector")
                return events
            
            # Extract data from each event element
            for i, element in enumerate(event_elements[:50], 1):  # Limit to 50 events
                try:
                    event_data = self._extract_event_data(element, i)
                    if event_data:
                        events.append(event_data)
                except Exception as e:
                    logger.debug(f"Error extracting event {i}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"❌ Error extracting calendar events: {e}")
        
        return events
    
    def _extract_event_data(self, element, event_number: int) -> Optional[Dict[str, Any]]:
        """Extract event data from a single event element"""
        try:
            # Get all text from the element
            element_text = element.text
            
            if not element_text or len(element_text) < 10:
                return None
            
            # Try to find title (usually in h3, h4, or a tag)
            title = None
            for tag in ['h3', 'h4', 'h2', 'a', '.title', '.event-title']:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, tag)
                    title = title_elem.text.strip()
                    if title and len(title) > 5:
                        break
                except NoSuchElementException:
                    continue
            
            # Fallback: use first line of text as title
            if not title:
                lines = [line.strip() for line in element_text.split('\n') if line.strip()]
                title = lines[0] if lines else None
            
            if not title or len(title) < 5:
                return None
            
            # Try to find date
            date_str = None
            date_selectors = ['.date', '.time', '.when', '[class*="date"]', '[class*="time"]']
            for selector in date_selectors:
                try:
                    date_elem = element.find_element(By.CSS_SELECTOR, selector)
                    date_str = date_elem.text.strip()
                    if date_str:
                        break
                except NoSuchElementException:
                    continue
            
            # Try to find venue
            venue = None
            venue_selectors = ['.venue', '.location', '.place', '[class*="venue"]', '[class*="location"]']
            for selector in venue_selectors:
                try:
                    venue_elem = element.find_element(By.CSS_SELECTOR, selector)
                    venue = venue_elem.text.strip()
                    if venue:
                        break
                except NoSuchElementException:
                    continue
            
            # Try to find event link
            event_url = None
            try:
                link = element.find_element(By.TAG_NAME, 'a')
                event_url = link.get_attribute('href')
            except NoSuchElementException:
                pass
            
            # Create event data
            event_data = {
                'title': title,
                'venue': venue or 'İKSV Venue',
                'date_str': date_str or 'TBA',
                'source': 'İKSV (Selenium)',
                'fetched_at': datetime.now().isoformat(),
                'event_number': event_number,
                'extraction_method': 'selenium'
            }
            
            if event_url:
                event_data['url'] = event_url
            
            # Add full text for additional context
            if len(element_text) < 500:
                event_data['full_text'] = element_text
            
            return event_data
            
        except Exception as e:
            logger.debug(f"Error parsing event element: {e}")
            return None
    
    def scrape_festival_page(self, url: str, festival_name: str) -> List[Dict[str, Any]]:
        """
        Scrape events from a specific festival page
        
        Args:
            url: Festival page URL
            festival_name: Name of the festival (e.g., "Jazz Festival")
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        try:
            logger.info(f"🌐 Loading {festival_name} page: {url}")
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Scroll to load content
            self._scroll_page()
            
            # Look for program/schedule sections
            program_selectors = [
                '.program', '.schedule', '.events',
                '[class*="program"]', '[class*="schedule"]',
                '.performances', '.concerts', '.shows'
            ]
            
            program_elements = []
            for selector in program_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        program_elements.extend(elements)
                except NoSuchElementException:
                    continue
            
            if not program_elements:
                logger.warning(f"⚠️ No program elements found on {festival_name} page")
                return events
            
            # Extract events from program sections
            for section in program_elements[:10]:  # Limit sections
                try:
                    section_events = self._extract_festival_events(section, festival_name)
                    events.extend(section_events)
                except Exception as e:
                    logger.debug(f"Error extracting from section: {e}")
                    continue
            
            logger.info(f"✅ Found {len(events)} events from {festival_name}")
            
        except Exception as e:
            logger.error(f"❌ Error scraping {festival_name}: {e}")
        
        return events
    
    def _extract_festival_events(self, section_element, festival_name: str) -> List[Dict[str, Any]]:
        """Extract events from a festival program section"""
        events = []
        
        try:
            # Find event items within section
            item_selectors = [
                '.event', '.show', '.concert', '.performance',
                '.item', '.card', '.listing'
            ]
            
            items = []
            for selector in item_selectors:
                try:
                    elements = section_element.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        items = elements
                        break
                except NoSuchElementException:
                    continue
            
            for i, item in enumerate(items[:30], 1):  # Limit to 30 per section
                event_data = self._extract_event_data(item, i)
                if event_data:
                    event_data['festival'] = festival_name
                    event_data['category'] = self._festival_to_category(festival_name)
                    events.append(event_data)
            
        except Exception as e:
            logger.debug(f"Error extracting festival events: {e}")
        
        return events
    
    def _festival_to_category(self, festival_name: str) -> str:
        """Map festival name to event category"""
        festival_lower = festival_name.lower()
        
        if 'jazz' in festival_lower or 'music' in festival_lower:
            return 'Music'
        elif 'film' in festival_lower or 'cinema' in festival_lower:
            return 'Film'
        elif 'theatre' in festival_lower or 'theater' in festival_lower:
            return 'Theatre'
        elif 'biennial' in festival_lower or 'art' in festival_lower:
            return 'Art'
        else:
            return 'Culture'
    
    def scrape_all_iksv_sources(self) -> List[Dict[str, Any]]:
        """
        Scrape events from all major İKSV sources
        
        Returns:
            Combined list of all events found
        """
        all_events = []
        
        # Scrape main calendar
        logger.info("🎯 Scraping main İKSV calendar...")
        main_events = self.scrape_main_calendar()
        all_events.extend(main_events)
        
        # Festival pages (only scrape if relevant)
        festival_pages = [
            ("https://caz.iksv.org/en", "Istanbul Jazz Festival"),
            ("https://film.iksv.org/en", "Istanbul Film Festival"),
            ("https://tiyatro.iksv.org/en", "Istanbul Theatre Festival"),
            ("https://muzik.iksv.org/en", "Istanbul Music Festival"),
        ]
        
        for url, name in festival_pages:
            try:
                logger.info(f"🎯 Scraping {name}...")
                festival_events = self.scrape_festival_page(url, name)
                all_events.extend(festival_events)
                time.sleep(2)  # Be polite between requests
            except Exception as e:
                logger.warning(f"⚠️ Could not scrape {name}: {e}")
                continue
        
        # Remove duplicates
        unique_events = self._remove_duplicates(all_events)
        
        logger.info(f"🎉 Total unique events scraped: {len(unique_events)}")
        
        return unique_events
    
    def _remove_duplicates(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate events based on title and date"""
        seen = set()
        unique_events = []
        
        for event in events:
            key = (event.get('title', '').lower(), event.get('date_str', ''))
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return unique_events


# Convenience function for one-off scraping
def scrape_iksv_events(headless: bool = True) -> List[Dict[str, Any]]:
    """
    Scrape İKSV events using Selenium (convenience function)
    
    Args:
        headless: Run browser in headless mode
        
    Returns:
        List of event dictionaries
    """
    with IKSVSeleniumScraper(headless=headless) as scraper:
        return scraper.scrape_all_iksv_sources()


# Test section
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎭 İKSV Selenium Scraper Test")
    print("=" * 60)
    
    try:
        with IKSVSeleniumScraper(headless=True) as scraper:
            # Test main calendar
            print("\n📅 Testing main calendar scraping...")
            events = scraper.scrape_main_calendar("https://www.iksv.org/en")
            
            print(f"\n✅ Found {len(events)} events!")
            
            if events:
                print("\n📋 Sample events:")
                for i, event in enumerate(events[:5], 1):
                    print(f"\n{i}. {event['title']}")
                    print(f"   📍 {event['venue']}")
                    print(f"   📅 {event['date_str']}")
            else:
                print("\n⚠️ No events found. This might be normal if:")
                print("   - The calendar is empty (off-season)")
                print("   - Page structure has changed")
                print("   - JavaScript didn't load properly")
                
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        print("\nMake sure Chrome browser is installed on your system.")
