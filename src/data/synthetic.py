"""
Synthetic data generator for testing and demonstration.
Generates realistic donors, campaigns, and donations.
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Sample data for generation
FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Oliver", "Amelia",
    "Benjamin", "Harper", "Elijah", "Evelyn", "Lucas", "Abigail", "Michael",
    "Emily", "Alexander", "Elizabeth", "Daniel", "Sofia", "Matthew", "Avery",
    "Aiden", "Ella", "Henry", "Scarlett", "Joseph", "Grace", "Jackson",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
]

CITIES = [
    ("New York", "NY", 40.7128, -74.0060),
    ("Los Angeles", "CA", 34.0522, -118.2437),
    ("Chicago", "IL", 41.8781, -87.6298),
    ("Houston", "TX", 29.7604, -95.3698),
    ("Phoenix", "AZ", 33.4484, -112.0740),
    ("Philadelphia", "PA", 39.9526, -75.1652),
    ("San Antonio", "TX", 29.4241, -98.4936),
    ("San Diego", "CA", 32.7157, -117.1611),
    ("Dallas", "TX", 32.7767, -96.7970),
    ("San Jose", "CA", 37.3382, -121.8863),
    ("Austin", "TX", 30.2672, -97.7431),
    ("Seattle", "WA", 47.6062, -122.3321),
    ("Denver", "CO", 39.7392, -104.9903),
    ("Boston", "MA", 42.3601, -71.0589),
    ("Nashville", "TN", 36.1627, -86.7816),
    ("Portland", "OR", 45.5152, -122.6784),
    ("Atlanta", "GA", 33.7490, -84.3880),
    ("Miami", "FL", 25.7617, -80.1918),
]

EMPLOYERS = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Uber",
    "Airbnb", "Salesforce", "Adobe", "Oracle", "IBM", "Intel", "Cisco",
    "Tesla", "SpaceX", "Twitter", "LinkedIn", "Stripe", "Square",
    "Local Hospital", "City School District", "State University",
    "Community Bank", "Regional Medical Center", "Tech Startup Inc",
]

SCHOOLS = [
    "Stanford University", "MIT", "Harvard University", "UC Berkeley",
    "Yale University", "Princeton University", "Columbia University",
    "University of Michigan", "UCLA", "NYU", "Duke University",
    "Northwestern University", "University of Texas", "Georgia Tech",
]

CAMPAIGN_TITLES_MEDICAL = [
    "Help {name} fight cancer",
    "Support {name}'s surgery recovery",
    "Medical expenses for {name}",
    "{name}'s chemotherapy fund",
    "Help {name} get the treatment they need",
    "Support {name}'s battle with {condition}",
    "{name} needs your help with medical bills",
    "Urgent: {name}'s transplant fund",
]

CAMPAIGN_TITLES_EDUCATION = [
    "Help {name} go to college",
    "Support {name}'s education",
    "{name}'s scholarship fund",
    "Send {name} to {school}",
    "Help {name} finish their degree",
    "{name}'s study abroad opportunity",
]

CAMPAIGN_TITLES_EMERGENCY = [
    "Help {name} rebuild after the fire",
    "Support {name} after the flood",
    "{name}'s family lost everything",
    "Emergency fund for {name}",
    "Help {name} after the accident",
]

CAMPAIGN_TITLES_COMMUNITY = [
    "Save our local {place}",
    "Help build a community {facility}",
    "Support the {neighborhood} cleanup",
    "Fund the local {program}",
]

MEDICAL_CONDITIONS = [
    "cancer", "leukemia", "heart disease", "diabetes complications",
    "rare genetic disorder", "kidney failure", "liver disease",
    "multiple sclerosis", "ALS", "Parkinson's disease",
]

COMMUNITY_PLACES = [
    "library", "park", "community center", "youth center",
    "animal shelter", "food bank", "homeless shelter",
]


class SyntheticDataGenerator:
    """Generates synthetic data for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self._generated_donors: Dict[str, Dict] = {}
        self._generated_campaigns: Dict[str, Dict] = {}
    
    def generate_donor(
        self,
        donor_id: Optional[str] = None,
        with_employer: bool = True,
        with_school: bool = True,
    ) -> Dict[str, Any]:
        """Generate a synthetic donor."""
        donor_id = donor_id or str(uuid4())
        
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        city, state, lat, lon = random.choice(CITIES)
        
        donor = {
            "donor_id": donor_id,
            "email": f"{first_name.lower()}.{last_name.lower()}@example.com",
            "display_name": f"{first_name} {last_name}",
            "first_name": first_name,
            "last_name": last_name,
            "city": city,
            "state": state,
            "country": "US",
            "coordinates": (lat + random.uniform(-0.1, 0.1), lon + random.uniform(-0.1, 0.1)),
            "joined_date": (datetime.now() - timedelta(days=random.randint(30, 1000))).isoformat(),
        }
        
        if with_employer and random.random() > 0.3:
            donor["employer"] = random.choice(EMPLOYERS)
            donor["employer_id"] = str(uuid4())
        
        if with_school and random.random() > 0.4:
            donor["schools"] = random.sample(SCHOOLS, k=random.randint(1, 2))
            donor["graduation_years"] = {
                school: random.randint(1990, 2023)
                for school in donor["schools"]
            }
        
        self._generated_donors[donor_id] = donor
        return donor
    
    def generate_campaign(
        self,
        campaign_id: Optional[str] = None,
        category: Optional[str] = None,
        organizer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a synthetic campaign."""
        campaign_id = campaign_id or str(uuid4())
        
        if not category:
            category = random.choice(["medical", "education", "emergency", "community"])
        
        # Generate beneficiary name
        beneficiary_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        city, state, lat, lon = random.choice(CITIES)
        
        # Generate title and description based on category
        if category == "medical":
            condition = random.choice(MEDICAL_CONDITIONS)
            title_template = random.choice(CAMPAIGN_TITLES_MEDICAL)
            title = title_template.format(name=beneficiary_name, condition=condition)
            description = self._generate_medical_description(beneficiary_name, condition)
            is_recurring = random.random() > 0.4  # 60% chance of recurring need
        elif category == "education":
            school = random.choice(SCHOOLS)
            title_template = random.choice(CAMPAIGN_TITLES_EDUCATION)
            title = title_template.format(name=beneficiary_name, school=school)
            description = self._generate_education_description(beneficiary_name, school)
            is_recurring = random.random() > 0.7
        elif category == "emergency":
            title_template = random.choice(CAMPAIGN_TITLES_EMERGENCY)
            title = title_template.format(name=beneficiary_name)
            description = self._generate_emergency_description(beneficiary_name)
            is_recurring = False
        else:  # community
            place = random.choice(COMMUNITY_PLACES)
            title_template = random.choice(CAMPAIGN_TITLES_COMMUNITY)
            title = title_template.format(place=place, facility=place, neighborhood="Downtown", program="youth")
            description = self._generate_community_description(place)
            is_recurring = random.random() > 0.5
        
        # Financial data
        goal_amount = random.choice([5000, 10000, 15000, 25000, 50000, 100000])
        funding_percentage = random.uniform(0.1, 0.95)
        raised_amount = round(goal_amount * funding_percentage, 2)
        donor_count = int(raised_amount / random.uniform(30, 100))
        
        # Create campaign
        created_at = datetime.now() - timedelta(days=random.randint(1, 180))
        
        campaign = {
            "campaign_id": campaign_id,
            "title": title,
            "description": description,
            "category": category,
            "city": city,
            "state": state,
            "country": "US",
            "coordinates": (lat + random.uniform(-0.1, 0.1), lon + random.uniform(-0.1, 0.1)),
            "goal_amount": goal_amount,
            "raised_amount": raised_amount,
            "donor_count": donor_count,
            "share_count": random.randint(10, 500),
            "comment_count": random.randint(5, 100),
            "organizer": {
                "id": organizer_id or str(uuid4()),
                "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                "verified": random.random() > 0.3,
                "campaign_count": random.randint(0, 5),
            },
            "updates": self._generate_updates(created_at),
            "created_at": created_at.isoformat(),
            "is_recurring_need": is_recurring,
            "beneficiary_name": beneficiary_name,
        }
        
        self._generated_campaigns[campaign_id] = campaign
        return campaign
    
    def generate_donation(
        self,
        donor_id: str,
        campaign_id: str,
        amount: Optional[float] = None,
        days_ago: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a synthetic donation."""
        if amount is None:
            # Generate realistic donation amount
            amount = random.choice([
                random.uniform(10, 25),
                random.uniform(25, 50),
                random.uniform(50, 100),
                random.uniform(100, 250),
                random.uniform(250, 500),
            ])
            amount = round(amount, 2)
        
        if days_ago is None:
            days_ago = random.randint(0, 365)
        
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        # Get campaign info for the donation
        campaign = self._generated_campaigns.get(campaign_id, {})
        
        return {
            "donation_id": str(uuid4()),
            "donor_id": donor_id,
            "campaign_id": campaign_id,
            "campaign_title": campaign.get("title", "Campaign"),
            "campaign_category": campaign.get("category", "other"),
            "campaign_description": campaign.get("description", "")[:500],
            "amount": amount,
            "timestamp": timestamp.isoformat(),
            "is_anonymous": random.random() > 0.85,
            "is_recurring": random.random() > 0.9,
            "source": random.choice(["direct", "shared", "search", "email", "social"]),
        }
    
    def generate_donor_with_history(
        self,
        num_donations: int = 10,
        categories: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate a donor with donation history."""
        donor = self.generate_donor()
        
        if categories is None:
            # Weight towards certain categories for more realistic affinity
            primary_category = random.choice(["medical", "education", "emergency", "community"])
            categories = [primary_category] * (num_donations // 2) + random.choices(
                ["medical", "education", "emergency", "community"],
                k=num_donations - num_donations // 2
            )
        
        donations = []
        for i, category in enumerate(categories[:num_donations]):
            # Generate or reuse campaign
            campaign = self.generate_campaign(category=category)
            
            # Generate donation with time spread
            donation = self.generate_donation(
                donor_id=donor["donor_id"],
                campaign_id=campaign["campaign_id"],
                days_ago=random.randint(i * 30, (i + 1) * 30 + 30),
            )
            donations.append(donation)
        
        return donor, donations
    
    def generate_dataset(
        self,
        num_donors: int = 100,
        num_campaigns: int = 200,
        avg_donations_per_donor: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a complete synthetic dataset."""
        donors = []
        campaigns = []
        donations = []
        
        # Generate campaigns first
        categories = ["medical", "education", "emergency", "community"]
        for _ in range(num_campaigns):
            campaign = self.generate_campaign(
                category=random.choice(categories)
            )
            campaigns.append(campaign)
        
        # Generate donors with donations
        for _ in range(num_donors):
            donor = self.generate_donor()
            donors.append(donor)
            
            # Generate donations for this donor
            num_donations = max(1, int(random.gauss(avg_donations_per_donor, 2)))
            
            for _ in range(num_donations):
                campaign = random.choice(campaigns)
                donation = self.generate_donation(
                    donor_id=donor["donor_id"],
                    campaign_id=campaign["campaign_id"],
                )
                donations.append(donation)
        
        return {
            "donors": donors,
            "campaigns": campaigns,
            "donations": donations,
        }
    
    def generate_social_connections(
        self,
        donors: List[Dict[str, Any]],
        avg_connections: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate social connections between donors."""
        connections = []
        donor_ids = [d["donor_id"] for d in donors]
        
        for donor in donors:
            num_connections = max(1, int(random.gauss(avg_connections, 2)))
            potential_connections = [d for d in donor_ids if d != donor["donor_id"]]
            
            if len(potential_connections) < num_connections:
                num_connections = len(potential_connections)
            
            connected_ids = random.sample(potential_connections, num_connections)
            
            for connected_id in connected_ids:
                connection_type = random.choice([
                    "friend", "family", "colleague", "classmate", "neighbor"
                ])
                
                connections.append({
                    "user_id": donor["donor_id"],
                    "connected_user_id": connected_id,
                    "connection_type": connection_type,
                    "strength": random.uniform(0.3, 1.0),
                    "is_mutual": random.random() > 0.2,
                })
        
        return connections
    
    def _generate_medical_description(self, name: str, condition: str) -> str:
        """Generate a medical campaign description."""
        templates = [
            f"{name} was recently diagnosed with {condition}. The medical bills are overwhelming, and we're reaching out to our community for help. Every dollar counts towards their treatment and recovery.",
            f"Our beloved {name} is fighting {condition}. The treatment costs are substantial, and insurance doesn't cover everything. Please help us support {name} through this difficult time.",
            f"{name} needs your help. After being diagnosed with {condition}, the family is facing significant medical expenses. Your generosity will help cover treatment costs and allow {name} to focus on getting better.",
        ]
        return random.choice(templates)
    
    def _generate_education_description(self, name: str, school: str) -> str:
        """Generate an education campaign description."""
        templates = [
            f"{name} has worked incredibly hard and has been accepted to {school}. However, the tuition costs are beyond what the family can afford. Help {name} achieve their dreams!",
            f"Help {name} pursue their education at {school}. Despite financial challenges, {name} has excelled academically and deserves this opportunity.",
            f"{name} is a first-generation college student heading to {school}. Your support will help cover tuition, books, and living expenses.",
        ]
        return random.choice(templates)
    
    def _generate_emergency_description(self, name: str) -> str:
        """Generate an emergency campaign description."""
        templates = [
            f"{name}'s family lost everything in a devastating house fire. They need immediate help with temporary housing, clothing, and basic necessities.",
            f"After a sudden accident, {name}'s family is facing unexpected expenses. Please help them get back on their feet during this difficult time.",
            f"{name} and their family were victims of a natural disaster. They need support to rebuild their lives and recover from this tragedy.",
        ]
        return random.choice(templates)
    
    def _generate_community_description(self, place: str) -> str:
        """Generate a community campaign description."""
        templates = [
            f"Our local {place} has been a cornerstone of our community for years. Now it needs renovations to continue serving our neighbors. Help us preserve this important resource!",
            f"We're raising funds to improve our community {place}. This project will benefit hundreds of families in our neighborhood.",
            f"Join us in supporting our local {place}. Your donation will help create a better space for everyone in our community.",
        ]
        return random.choice(templates)
    
    def _generate_updates(self, created_at: datetime) -> List[Dict[str, Any]]:
        """Generate campaign updates."""
        num_updates = random.randint(0, 5)
        updates = []
        
        update_types = ["progress", "milestone", "gratitude"]
        update_templates = {
            "progress": [
                "Thank you for your continued support! We're making progress.",
                "Update: Treatment is going well. Your support means everything.",
                "We've reached {percent}% of our goal! Keep sharing!",
            ],
            "milestone": [
                "Amazing! We've reached ${amount} raised!",
                "Milestone achieved! Thank you to all {count} donors!",
                "We did it! Goal reached thanks to your generosity!",
            ],
            "gratitude": [
                "From the bottom of our hearts, thank you.",
                "Your kindness has touched our family deeply.",
                "We are overwhelmed by the support from this community.",
            ],
        }
        
        for i in range(num_updates):
            update_type = random.choice(update_types)
            template = random.choice(update_templates[update_type])
            
            content = template.format(
                percent=random.randint(25, 100),
                amount=random.randint(1000, 50000),
                count=random.randint(10, 500),
            )
            
            update_date = created_at + timedelta(days=random.randint(1, 60) * (i + 1))
            
            updates.append({
                "date": update_date.isoformat(),
                "content": content,
                "type": update_type,
            })
        
        return updates


# Convenience function
def generate_demo_data() -> Dict[str, Any]:
    """Generate a demo dataset for testing."""
    generator = SyntheticDataGenerator(seed=42)
    
    dataset = generator.generate_dataset(
        num_donors=50,
        num_campaigns=100,
        avg_donations_per_donor=8,
    )
    
    # Add social connections
    dataset["connections"] = generator.generate_social_connections(
        dataset["donors"],
        avg_connections=4,
    )
    
    return dataset





