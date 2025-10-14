#!/usr/bin/env python3
"""
Museum Information Integration for Istanbul AI
==============================================

This script integrates the updated museum database with the AI system,
removing price information and adding MuseumPass details.
"""

from updated_museum_database import UpdatedIstanbulMuseumDatabase, MuseumInfo
from google_maps_hours_checker import GoogleMapsHoursChecker
from typing import Dict, List, Optional

class MuseumResponseGenerator:
    """Generate museum responses for the AI system"""
    
    def __init__(self):
        self.museum_db = UpdatedIstanbulMuseumDatabase()
        self.hours_checker = GoogleMapsHoursChecker()
    
    def generate_museum_recommendation(self, query: str, context: str = "") -> str:
        """Generate museum recommendation based on query"""
        
        # Determine query type
        if "free" in query.lower() or "budget" in query.lower():
            return self._generate_budget_friendly_response()
        elif "art" in query.lower() or "cultural" in query.lower():
            return self._generate_cultural_museums_response()
        elif "historical" in query.lower() or "history" in query.lower():
            return self._generate_historical_museums_response()
        elif "museumpass" in query.lower() or "museum pass" in query.lower():
            return self._generate_museumpass_info()
        else:
            return self._generate_general_museums_response()
    
    def _generate_budget_friendly_response(self) -> str:
        """Generate response for budget-conscious visitors"""
        response = """
I totally understand wanting to explore Istanbul without breaking the bank! 😊 The good news is there are plenty of amazing places you can visit for free, plus some smart ways to save money on the paid attractions.

**Let's start with the completely FREE gems:**
🕌 **Hagia Sophia** is absolutely stunning and won't cost you a penny since it's now a functioning mosque again. Just remember to visit between prayer times and dress modestly - it's respectful and you'll get the full experience.

🕌 **Blue Mosque** is another must-see that's totally free. The blue tiles inside are breathtaking, especially when the light hits them just right in the afternoon.

🛍️ **Grand Bazaar** and **Spice Bazaar** are like stepping into a time machine - and browsing is completely free! Even if you don't buy anything, the atmosphere alone is worth the visit. The colors, smells, and energy are incredible.

**Now, for the museums that do charge entry fees...**
Here's where I'll share a little insider secret: if you're planning to visit 3 or more museums, the **MuseumPass Istanbul** (valid for 5 days) is absolutely worth getting! It saves you money AND lets you skip some lines! 

It gets you into amazing places like:
• Topkapi Palace (where sultans actually lived!)
• Istanbul Archaeological Museums (that Alexander Sarcophagus is mind-blowing)
• Museum of Turkish and Islamic Arts (the carpet collection is gorgeous)
• Galata Tower (those views... wow!)

**My budget-friendly tips from experience:**
💡 Always carry your student ID if you have one - many places offer discounts
💡 The neighborhoods of Balat and Fener are like open-air museums and completely free to wander
💡 Sunset at Galata Bridge costs nothing but gives you million-dollar views

Want me to tell you more about any of these places? I'm happy to help you plan the perfect budget-friendly Istanbul adventure! 🌟
"""
        return response.strip()
    
    def _generate_cultural_museums_response(self) -> str:
        """Generate response for cultural/art museums"""
        cultural_museums = [
            self.museum_db.get_museum_info('museum_turkish_islamic_arts'),
            self.museum_db.get_museum_info('istanbul_archaeological_museums')
        ]
        
        response = """
Oh, you're in for such a treat! Istanbul's cultural museums are like treasure chests filled with centuries of art and history. Let me share my favorites with you! 🎨

🏛️ **Museum of Turkish and Islamic Arts** 
This place is housed in what used to be Ibrahim Pasha's palace - can you imagine living there?! It's right on Sultanahmet Square, so super convenient.

The carpet collection here will absolutely blow your mind - some of these pieces are so intricate you'll wonder how human hands created them. And the Islamic manuscripts... the calligraphy is pure poetry made visible. 

*Hours: Usually 9am-5pm (closed Mondays) | MuseumPass accepted ✅*
*Perfect for: 1-2 hours of pure cultural immersion*

🏛️ **Istanbul Archaeological Museums** 
Okay, this is where I get really excited! This isn't just one museum - it's actually three buildings packed with incredible finds. The Alexander Sarcophagus is the star here, and when you see it in person, you'll understand why archaeologists get so passionate about their work.

But here's what I love most - they have the Treaty of Kadesh, which is literally the world's oldest known peace treaty. Standing in front of something that significant... it gives you chills in the best way.

*Hours: 9am-5pm (closed Mondays) | MuseumPass accepted ✅*
*Perfect for: 2-3 hours if you really want to soak it all in*

**Here's my insider tip:** Both of these accept the MuseumPass, and if you're planning to visit a few museums, it's honestly such a good deal. Plus, you get that satisfying feeling of just walking in without fumbling for tickets! 

Would you like me to tell you more about what makes these places so special? Or are you curious about planning the perfect cultural day in Istanbul? 😊
"""
        return response.strip()
    
    def _generate_historical_museums_response(self) -> str:
        """Generate response for historical museums"""
        historical_museums = [
            self.museum_db.get_museum_info('topkapi_palace'),
            self.museum_db.get_museum_info('istanbul_archaeological_museums')
        ]
        
        response = """
History lover? You've come to the right place! Istanbul's historical museums are like time machines that transport you back centuries. Let me tell you about the ones that will absolutely captivate you! ⏳

🏰 **Topkapi Palace Museum**
This is where the magic of the Ottoman Empire comes alive! For 400 years, sultans called this place home, and walking through it today, you can almost hear the whispers of palace intrigue and royal ceremonies.

The Harem is absolutely fascinating - it's like stepping into the private world of the Ottoman royal family. And the Treasury? Prepare to be dazzled by jewels that would make any crown envious! The Sacred Relics collection is deeply moving too - it's where some of Islam's most precious artifacts are kept.

*What I love most:* Standing in the same courtyards where sultans once walked and looking out at the same Bosphorus views they enjoyed every day.

*Practical stuff:* Plan for 2-4 hours (trust me, you'll want to linger), closes Tuesdays, and yes - MuseumPass works here! 🎫✅

🏺 **Istanbul Archaeological Museums**
This place is a history buff's paradise! Built back in 1891, it was Turkey's very first archaeological museum, and they've been collecting treasures ever since.

The Alexander Sarcophagus will leave you speechless - the craftsmanship is so detailed you can see individual expressions on the carved faces. And get this - they have the Treaty of Kadesh, the world's oldest peace treaty! It's incredible to think you're looking at humanity's first attempt at diplomacy written in stone.

*Pro tip:* The three buildings can be overwhelming, so I'd suggest starting with the main Archaeological Museum first - that's where the real showstoppers are.

*Time to visit:* Give yourself 2-3 hours, closed Mondays, and your MuseumPass will work here too! 🎫✅

**Between you and me:** These two museums alone are worth getting the MuseumPass for. The stories these places tell... they'll stay with you long after you leave Istanbul.

Want me to share some secret spots within these museums that most tourists miss? 😉
"""
        return response.strip()
    
    def _generate_museumpass_info(self) -> str:
        """Generate detailed MuseumPass information"""
        museumpass_museums = self.museum_db.get_museum_pass_museums()
        
        response = """
Ah, the MuseumPass Istanbul! This little card is honestly one of the smartest investments you can make for your Istanbul adventure. Let me break it down for you in a way that actually makes sense! 🎫

**So what's the deal exactly?**
You get a magic pass that opens the doors to 13 amazing museums for 5 whole days. But here's the thing - it's not just about saving money (though you definitely will if you visit 3+ museums). It's about the convenience! No more standing in ticket lines or fumbling for exact change.

**Here's how it works:**
Once you use it at your first museum, your 5-day countdown begins. You can visit each museum once - which honestly is perfect because there's so much to see in Istanbul, you won't want to repeat anyway!

**The museums you'll fall in love with:**
"""
        
        for museum in museumpass_museums:
            if "Archaeological" in museum.name:
                response += f"• **{museum.name}** - Home to that incredible Alexander Sarcophagus I mentioned!\n"
            elif "Topkapi" in museum.name:
                response += f"• **{museum.name}** - The sultan's palace with all its secrets and treasures\n"
            elif "Turkish and Islamic" in museum.name:
                response += f"• **{museum.name}** - Those stunning carpets and manuscripts\n"
            elif "Galata Tower" in museum.name:
                response += f"• **{museum.name}** - For those Instagram-worthy panoramic views\n"
            else:
                response += f"• **{museum.name}**\n"
        
        response += """
**A few friendly heads-ups:**
🕐 Some museums have time restrictions with the pass - like Galata Tower (entry by 6:14 PM) and a few others (by 6:45 PM). Nothing too crazy, just plan accordingly!

🌙 The pass doesn't work for special night programs after 7 PM - but honestly, the regular hours give you plenty of time to explore.

**My honest opinion?** If you're the type who loves museums and culture (which you clearly are since you're asking!), this pass pays for itself quickly and makes your trip so much smoother. Plus, there's something satisfying about just flashing your pass and walking right in! 

Want me to suggest the perfect 3-day itinerary using your MuseumPass? I've got some ideas that'll blow your mind! 😊
"""
        return response.strip()
    
    def _generate_general_museums_response(self) -> str:
        """Generate general museum recommendations"""
        all_museums = list(self.museum_db.get_all_museums().values())[:4]  # Top 4
        
        response = """
Perfect question! Istanbul has so many incredible museums that choosing can feel overwhelming, so let me share my absolute must-sees with you! These are the ones that will give you the best taste of this amazing city's rich history and culture. 🌟

"""
        
        for i, museum in enumerate(all_museums, 1):
            hours_info = self.hours_checker.get_formatted_hours(museum.name)
            
            if "Archaeological" in museum.name:
                museum_story = "This is where you'll find treasures that archaeologists dream about! The Alexander Sarcophagus alone is worth the visit - the detail is so incredible you'll spend ages just staring at it."
            elif "Topkapi" in museum.name:
                museum_story = "The former home of Ottoman sultans! Walking through these rooms, you'll feel like you've stepped into a real-life Arabian Nights tale. The Treasury will make your jaw drop."
            elif "Turkish and Islamic" in museum.name:
                museum_story = "Housed in a beautiful Ottoman palace, this place showcases the artistic soul of Turkish and Islamic culture. The carpet collection is absolutely mesmerizing!"
            elif "Galata Tower" in museum.name:
                museum_story = "Not just a museum, but the best viewpoint in the city! On a clear day, you can see all the way across the Bosphorus. Perfect for those Instagram shots too! 📸"
            else:
                museum_story = "A hidden gem that most visitors overlook, but shouldn't!"
            
            response += f"""
**{i}. {museum.name}**
{museum_story}

📍 *Where to find it:* {museum.location}
⏰ *When to go:* {hours_info.get('daily_summary', 'Check current hours')}
🎫 *MuseumPass friendly?* {'Yes! ' if museum.museum_pass_valid else 'Nope, separate ticket needed'}
⏱️ *Time needed:* {museum.visiting_duration}

"""
        
        response += """
**Here's my insider advice:** If you're planning to visit more than 2-3 museums, definitely grab the MuseumPass (valid for 5 days). It's not just about convenience - though you'll love just walking up and getting in without the ticket hassle - it's also a smart way to explore more places without worrying about individual entry fees.

And here's a little secret: visit the museums in the morning when possible. They're less crowded, the light is beautiful for photos, and you'll have more energy to really appreciate what you're seeing.

Which of these sounds most interesting to you? I'd love to tell you more about whichever ones catch your eye! 😊
"""
        return response.strip()

def test_museum_responses():
    """Test different types of museum responses"""
    generator = MuseumResponseGenerator()
    
    test_queries = [
        "Show me budget-friendly museums and free attractions",
        "What are the best art and cultural museums?", 
        "Tell me about historical museums",
        "What is MuseumPass Istanbul?",
        "Recommend some museums to visit"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        response = generator.generate_museum_recommendation(query)
        print(response[:500] + "...\n")

if __name__ == "__main__":
    test_museum_responses()
