"""Synthesis and writing tasks.

Six factory functions producing RubricDatapoint instances for document
synthesis / business writing problems. Each gives the agent source material
in necessary_files and asks it to produce a polished deliverable.

Factories:
  make_meeting_minutes          (seedable)
  make_customer_complaint_response (seedable)
  make_competitive_comparison   (seedable)
  make_press_release            (seedable)
  make_literature_synthesis     (static)
  make_budget_allocation        (seedable)
"""

import random as _random
from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    make_name,
    make_names,
    pick_one,
    pick,
    vary_int,
    COMPANY_NAMES,
)

# =============================================================================
# DOMAIN: PRODUCTS (for comparison, press release, etc.)
# =============================================================================

PRODUCTS = [
    {
        "name": "CloudSync Pro 3.0",
        "category": "Cloud Storage",
        "specs": {"storage": "5 TB", "max_file_size": "50 GB", "sync_speed": "500 Mbps",
                  "encryption": "AES-256", "platforms": "Windows, Mac, Linux, iOS, Android",
                  "price": "$12.99/month", "users": "Up to 10", "api_access": "Yes"},
    },
    {
        "name": "DataVault Enterprise",
        "category": "Cloud Storage",
        "specs": {"storage": "Unlimited", "max_file_size": "100 GB", "sync_speed": "1 Gbps",
                  "encryption": "AES-256 + at-rest", "platforms": "Windows, Mac, Linux",
                  "price": "$29.99/month", "users": "Unlimited", "api_access": "Yes"},
    },
    {
        "name": "SimpleStore Basic",
        "category": "Cloud Storage",
        "specs": {"storage": "2 TB", "max_file_size": "25 GB", "sync_speed": "200 Mbps",
                  "encryption": "AES-128", "platforms": "Windows, Mac, iOS, Android",
                  "price": "$4.99/month", "users": "Up to 3", "api_access": "No"},
    },
    {
        "name": "ThermoGuard X500",
        "category": "Smart Thermostat",
        "specs": {"display": "4.3 inch color touchscreen", "connectivity": "WiFi 6, Bluetooth 5.2, Zigbee",
                  "sensors": "Temperature, humidity, occupancy, ambient light",
                  "compatibility": "Alexa, Google Home, Apple HomeKit, IFTTT",
                  "price": "$249", "warranty": "3 years", "energy_savings": "Up to 23%"},
    },
    {
        "name": "EcoTemp Smart",
        "category": "Smart Thermostat",
        "specs": {"display": "3.5 inch LCD", "connectivity": "WiFi 5, Bluetooth 5.0",
                  "sensors": "Temperature, humidity",
                  "compatibility": "Alexa, Google Home",
                  "price": "$129", "warranty": "2 years", "energy_savings": "Up to 15%"},
    },
    {
        "name": "ClimateControl Pro",
        "category": "Smart Thermostat",
        "specs": {"display": "5 inch OLED touchscreen", "connectivity": "WiFi 6E, Bluetooth 5.3, Thread, Matter",
                  "sensors": "Temperature, humidity, occupancy, air quality (PM2.5), CO2",
                  "compatibility": "Alexa, Google Home, Apple HomeKit, Samsung SmartThings, IFTTT",
                  "price": "$349", "warranty": "5 years", "energy_savings": "Up to 30%"},
    },
]


# ============================================================================
# POOL DATA (local to this module)
# ============================================================================

MEETING_TOPICS = [
    "Q3 Marketing Strategy Planning",
    "Product Roadmap Review for FY2025",
    "Annual Budget Allocation Discussion",
    "Customer Success Initiative Kickoff",
    "Engineering Infrastructure Migration",
    "New Hire Onboarding Process Redesign",
    "Quarterly OKR Review and Alignment",
    "Data Privacy Compliance Update",
    "Partnership Strategy with Channel Distributors",
    "Office Relocation Planning",
]

COMPLAINT_ISSUES = [
    {
        "type": "wrong_size",
        "product": "UltraFit Running Shoes",
        "sku": "UF-RUN-2024-M10",
        "complaint_detail": (
            "I ordered size 10 but received size 8. The box label says size 10 "
            "but the shoes inside are clearly size 8. I measured them and they "
            "are 26.0 cm, which matches your size 8 chart."
        ),
        "product_specs": (
            "Product: UltraFit Running Shoes (UF-RUN-2024)\n"
            "Category: Athletic Footwear\n"
            "Available sizes: 6-14 (US Men's)\n"
            "Size 8: 26.0 cm insole length\n"
            "Size 10: 28.0 cm insole length\n"
            "Weight: 280g (size 10)\n"
            "Material: Engineered mesh upper, EVA midsole, rubber outsole\n"
            "Price: $129.99\n"
            "Warranty: 90-day manufacturing defect warranty\n"
        ),
        "return_window": 30,
        "remedy": "free exchange with expedited shipping",
    },
    {
        "type": "defective",
        "product": "AquaPure Water Filter Pitcher",
        "sku": "AP-WFP-3000",
        "complaint_detail": (
            "The filter cartridge leaks unfiltered water through the side "
            "seal. I've tried reseating the cartridge three times following "
            "your YouTube tutorial. There is a visible gap between the "
            "cartridge gasket and the pitcher housing."
        ),
        "product_specs": (
            "Product: AquaPure Water Filter Pitcher (AP-WFP-3000)\n"
            "Category: Kitchen Appliances\n"
            "Capacity: 10-cup (2.4L filtered)\n"
            "Filter life: 40 gallons / ~2 months\n"
            "Filtration: 5-stage activated carbon + ion exchange\n"
            "Certifications: NSF 42, NSF 53\n"
            "Dimensions: 10.5 x 5.5 x 10.2 inches\n"
            "Price: $34.99 (pitcher), $14.99 (replacement filter)\n"
            "Warranty: 1-year limited warranty covering defects in "
            "materials and workmanship\n"
        ),
        "return_window": 365,
        "remedy": "full replacement unit with new filter cartridge",
    },
    {
        "type": "late_delivery",
        "product": "ProDesk Standing Desk Converter",
        "sku": "PD-SDC-4228",
        "complaint_detail": (
            "I ordered this desk converter on November 15 with guaranteed "
            "delivery by November 22 for my home office setup. It arrived "
            "on December 3 -- 11 days late. I had to purchase a temporary "
            "solution from a local store for $89 because I needed it for "
            "a remote work deadline."
        ),
        "product_specs": (
            "Product: ProDesk Standing Desk Converter (PD-SDC-4228)\n"
            "Category: Office Furniture\n"
            "Height range: 6.5 - 17 inches\n"
            "Desktop size: 42 x 28 inches\n"
            "Weight capacity: 35 lbs\n"
            "Weight: 52 lbs\n"
            "Assembly: Tool-free, ~15 minutes\n"
            "Shipping: Standard (5-7 business days), Expedited (2-3 days)\n"
            "Price: $299.99\n"
            "Warranty: 5-year structural warranty\n"
        ),
        "return_window": 30,
        "remedy": "full refund of shipping charges and $25 store credit",
    },
    {
        "type": "missing_parts",
        "product": "BuildRight Modular Bookshelf",
        "sku": "BR-MBS-6T-WAL",
        "complaint_detail": (
            "The box arrived with only 4 of the 6 shelf panels. The hardware "
            "bag is complete and the instructions reference 6 panels (parts "
            "P1 through P6) but P3 and P5 are missing. I cannot assemble the "
            "unit without them."
        ),
        "product_specs": (
            "Product: BuildRight Modular Bookshelf, 6-Tier (BR-MBS-6T-WAL)\n"
            "Category: Home Furniture\n"
            "Material: Walnut veneer engineered wood\n"
            "Dimensions: 72H x 30W x 12D inches\n"
            "Shelf panels: 6 (P1-P6), each 29.5 x 11.5 x 0.75 inches\n"
            "Hardware kit: 24 cam bolts, 24 cam locks, 12 shelf pins, "
            "8 wall-mount anchors\n"
            "Weight capacity per shelf: 30 lbs\n"
            "Price: $179.99\n"
            "Warranty: 2-year warranty on parts and structural integrity\n"
        ),
        "return_window": 60,
        "remedy": "ship missing panels with expedited delivery",
    },
]

SOFTWARE_PRODUCT_SETS = [
    {
        "category": "Project Management Software",
        "products": [
            {
                "name": "TaskFlow Pro",
                "price": "$15/user/month",
                "storage": "50 GB",
                "max_users": "Unlimited",
                "api_calls": "10,000/month",
                "uptime_sla": "99.9%",
                "support": "24/7 phone + chat",
                "integrations": "200+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "ProjectHub",
                "price": "$9/user/month",
                "storage": "100 GB",
                "max_users": "Up to 50",
                "api_calls": "5,000/month",
                "uptime_sla": "99.5%",
                "support": "Business hours email",
                "integrations": "75+",
                "mobile_app": "iOS only",
            },
            {
                "name": "WorkStream Enterprise",
                "price": "$29/user/month",
                "storage": "Unlimited",
                "max_users": "Unlimited",
                "api_calls": "Unlimited",
                "uptime_sla": "99.99%",
                "support": "24/7 phone + chat + dedicated CSM",
                "integrations": "500+",
                "mobile_app": "iOS, Android, Web PWA",
            },
        ],
        "best_at": {
            "TaskFlow Pro": "support and mobile availability",
            "ProjectHub": "storage per dollar and affordability",
            "WorkStream Enterprise": "uptime, integrations, and scalability",
        },
    },
    {
        "category": "Customer Relationship Management (CRM)",
        "products": [
            {
                "name": "SalesForce Lite",
                "price": "$25/user/month",
                "storage": "10 GB per user",
                "max_users": "Up to 100",
                "api_calls": "25,000/day",
                "uptime_sla": "99.95%",
                "support": "24/7 phone",
                "integrations": "300+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "RelateIQ",
                "price": "$12/user/month",
                "storage": "5 GB per user",
                "max_users": "Up to 25",
                "api_calls": "5,000/day",
                "uptime_sla": "99.5%",
                "support": "Business hours chat",
                "integrations": "50+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "PipelinePro",
                "price": "$45/user/month",
                "storage": "Unlimited",
                "max_users": "Unlimited",
                "api_calls": "Unlimited",
                "uptime_sla": "99.99%",
                "support": "24/7 phone + dedicated account team",
                "integrations": "800+",
                "mobile_app": "iOS, Android, Desktop (Win/Mac)",
            },
        ],
        "best_at": {
            "SalesForce Lite": "API volume and mid-tier value",
            "RelateIQ": "price point and ease of adoption",
            "PipelinePro": "scalability, uptime, and integrations",
        },
    },
    {
        "category": "Cloud Analytics Platform",
        "products": [
            {
                "name": "DataLens",
                "price": "$200/month (flat)",
                "storage": "500 GB",
                "max_users": "Up to 20",
                "api_calls": "50,000/month",
                "uptime_sla": "99.9%",
                "support": "Business hours email + chat",
                "integrations": "120+",
                "mobile_app": "Web only",
            },
            {
                "name": "InsightEngine",
                "price": "$350/month (flat)",
                "storage": "2 TB",
                "max_users": "Up to 50",
                "api_calls": "200,000/month",
                "uptime_sla": "99.95%",
                "support": "24/7 chat + email",
                "integrations": "250+",
                "mobile_app": "iOS, Android, Web",
            },
            {
                "name": "AnalyticsForge",
                "price": "$99/month (flat)",
                "storage": "100 GB",
                "max_users": "Up to 5",
                "api_calls": "10,000/month",
                "uptime_sla": "99.0%",
                "support": "Community forum + email",
                "integrations": "40+",
                "mobile_app": "Web only",
            },
        ],
        "best_at": {
            "DataLens": "value per GB and mid-tier reliability",
            "InsightEngine": "storage, API volume, and mobile access",
            "AnalyticsForge": "lowest cost for small teams",
        },
    },
    {
        "category": "Team Communication Platform",
        "products": [
            {
                "name": "ChatSync",
                "price": "$8/user/month",
                "storage": "20 GB",
                "max_users": "Up to 100",
                "api_calls": "5,000/month",
                "uptime_sla": "99.9%",
                "support": "Business hours email",
                "integrations": "150+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "TeamBridge Pro",
                "price": "$18/user/month",
                "storage": "100 GB",
                "max_users": "Unlimited",
                "api_calls": "50,000/month",
                "uptime_sla": "99.99%",
                "support": "24/7 phone + chat",
                "integrations": "400+",
                "mobile_app": "iOS, Android, Desktop (Win/Mac/Linux)",
            },
            {
                "name": "Huddle",
                "price": "$4/user/month",
                "storage": "5 GB",
                "max_users": "Up to 25",
                "api_calls": "1,000/month",
                "uptime_sla": "99.0%",
                "support": "Community forum",
                "integrations": "30+",
                "mobile_app": "iOS only",
            },
        ],
        "best_at": {
            "ChatSync": "mid-tier balance of features and price",
            "TeamBridge Pro": "enterprise scalability and reliability",
            "Huddle": "affordability for very small teams",
        },
    },
    {
        "category": "Email Marketing Automation",
        "products": [
            {
                "name": "MailJet Express",
                "price": "$29/month (up to 5,000 contacts)",
                "storage": "10 GB",
                "max_users": "Up to 3",
                "api_calls": "Unlimited",
                "uptime_sla": "99.5%",
                "support": "Business hours chat",
                "integrations": "80+",
                "mobile_app": "Web only",
            },
            {
                "name": "CampaignForge",
                "price": "$79/month (up to 25,000 contacts)",
                "storage": "50 GB",
                "max_users": "Up to 10",
                "api_calls": "Unlimited",
                "uptime_sla": "99.95%",
                "support": "24/7 phone + chat",
                "integrations": "250+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "SendPulse Lite",
                "price": "$15/month (up to 2,500 contacts)",
                "storage": "5 GB",
                "max_users": "1",
                "api_calls": "10,000/month",
                "uptime_sla": "99.0%",
                "support": "Email only",
                "integrations": "40+",
                "mobile_app": "Web only",
            },
        ],
        "best_at": {
            "MailJet Express": "unlimited API calls at mid-tier pricing",
            "CampaignForge": "largest contact list, integrations, and support",
            "SendPulse Lite": "lowest entry cost for solopreneurs",
        },
    },
    {
        "category": "Help Desk / Ticketing System",
        "products": [
            {
                "name": "TicketFlow",
                "price": "$19/agent/month",
                "storage": "25 GB",
                "max_users": "Up to 50 agents",
                "api_calls": "20,000/month",
                "uptime_sla": "99.9%",
                "support": "24/7 email + chat",
                "integrations": "120+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "SupportDesk Enterprise",
                "price": "$49/agent/month",
                "storage": "Unlimited",
                "max_users": "Unlimited",
                "api_calls": "Unlimited",
                "uptime_sla": "99.99%",
                "support": "24/7 phone + dedicated CSM",
                "integrations": "500+",
                "mobile_app": "iOS, Android, Desktop",
            },
            {
                "name": "HelpNow Basic",
                "price": "$9/agent/month",
                "storage": "5 GB",
                "max_users": "Up to 10 agents",
                "api_calls": "5,000/month",
                "uptime_sla": "99.5%",
                "support": "Business hours email",
                "integrations": "50+",
                "mobile_app": "Web only",
            },
        ],
        "best_at": {
            "TicketFlow": "mid-market value with mobile and 24/7 support",
            "SupportDesk Enterprise": "unlimited everything for large teams",
            "HelpNow Basic": "cheapest option for small support teams",
        },
    },
    {
        "category": "Document Collaboration Suite",
        "products": [
            {
                "name": "DocuShare",
                "price": "$12/user/month",
                "storage": "100 GB",
                "max_users": "Unlimited",
                "api_calls": "15,000/month",
                "uptime_sla": "99.9%",
                "support": "Business hours chat + email",
                "integrations": "200+",
                "mobile_app": "iOS, Android, Web",
            },
            {
                "name": "PaperTrail Pro",
                "price": "$6/user/month",
                "storage": "25 GB",
                "max_users": "Up to 20",
                "api_calls": "5,000/month",
                "uptime_sla": "99.5%",
                "support": "Email only",
                "integrations": "60+",
                "mobile_app": "Web only",
            },
            {
                "name": "CollabVault Enterprise",
                "price": "$25/user/month",
                "storage": "Unlimited",
                "max_users": "Unlimited",
                "api_calls": "Unlimited",
                "uptime_sla": "99.99%",
                "support": "24/7 phone + dedicated team",
                "integrations": "600+",
                "mobile_app": "iOS, Android, Desktop (Win/Mac)",
            },
        ],
        "best_at": {
            "DocuShare": "best storage-per-dollar with broad mobile support",
            "PaperTrail Pro": "budget-friendly for small teams",
            "CollabVault Enterprise": "unlimited scale and premium support",
        },
    },
    {
        "category": "Code Repository and CI/CD Platform",
        "products": [
            {
                "name": "CodeVault",
                "price": "$20/user/month",
                "storage": "50 GB",
                "max_users": "Unlimited",
                "api_calls": "30,000/month",
                "uptime_sla": "99.95%",
                "support": "24/7 chat + email",
                "integrations": "300+",
                "mobile_app": "iOS, Android",
            },
            {
                "name": "GitForge",
                "price": "$7/user/month",
                "storage": "10 GB",
                "max_users": "Up to 15",
                "api_calls": "5,000/month",
                "uptime_sla": "99.5%",
                "support": "Community forum + email",
                "integrations": "80+",
                "mobile_app": "Web only",
            },
            {
                "name": "DevOps Central",
                "price": "$40/user/month",
                "storage": "Unlimited",
                "max_users": "Unlimited",
                "api_calls": "Unlimited",
                "uptime_sla": "99.99%",
                "support": "24/7 phone + dedicated engineer",
                "integrations": "700+",
                "mobile_app": "iOS, Android, Desktop, VS Code extension",
            },
        ],
        "best_at": {
            "CodeVault": "strong mid-tier with good API and mobile",
            "GitForge": "lowest cost for small dev teams",
            "DevOps Central": "full enterprise DevOps with premium tooling",
        },
    },
]

PRESS_RELEASE_PRODUCTS = [
    {
        "product_name": "SmartHive Collaboration Platform 4.0",
        "launch_date": "March 15, 2025",
        "tagline": "Where ideas meet execution",
        "key_features": [
            "AI-powered meeting summarization with action item extraction",
            "Cross-platform whiteboard with real-time collaboration for up to 500 users",
            "Integrated project timeline synced to team calendars",
            "End-to-end encryption with SOC 2 Type II compliance",
        ],
        "price": "Starting at $8/user/month for teams of 10+",
        "availability": "General availability worldwide, enterprise beta in APAC",
    },
    {
        "product_name": "EcoTrack Carbon Footprint Manager",
        "launch_date": "April 22, 2025",
        "tagline": "Measure. Reduce. Report.",
        "key_features": [
            "Automated Scope 1, 2, and 3 emissions tracking",
            "Integration with 40+ ERP and supply chain platforms",
            "Regulatory-ready reports for SEC, EU CSRD, and ISSB frameworks",
            "Predictive modeling for net-zero pathway planning",
        ],
        "price": "Custom pricing based on company size; free tier for <50 employees",
        "availability": "North America and EU at launch, APAC Q3 2025",
    },
    {
        "product_name": "MediLink Patient Portal 2.0",
        "launch_date": "June 1, 2025",
        "tagline": "Healthcare, connected",
        "key_features": [
            "HIPAA-compliant video consultations with screen-share",
            "AI triage chatbot with symptom assessment and appointment routing",
            "Unified patient record view across participating providers",
            "Prescription refill and lab result notifications",
        ],
        "price": "$3,500/month per practice (up to 20 providers)",
        "availability": "US healthcare systems, pilot program for UK NHS trusts",
    },
    {
        "product_name": "QuantumShield Endpoint Security Suite",
        "launch_date": "February 28, 2025",
        "tagline": "Post-quantum protection, today",
        "key_features": [
            "Hybrid classical/post-quantum cryptographic key exchange (CRYSTALS-Kyber)",
            "Zero-trust network access with continuous device posture assessment",
            "Behavioral anomaly detection with <200ms response time",
            "Unified agent for Windows, macOS, Linux, iOS, and Android",
        ],
        "price": "$18/endpoint/month, volume discounts available",
        "availability": "Global availability with FedRAMP authorization pending",
    },
]

STYLE_GUIDE_TEMPLATES = [
    {
        "voice": "professional and forward-looking",
        "word_limit": 600,
        "required_sections": [
            "Headline",
            "Subheadline",
            "Dateline",
            "Lead paragraph",
            "Product details",
            "Executive quote",
            "Availability and pricing",
            "Boilerplate",
            "Media contact",
        ],
        "rules": [
            "Use active voice throughout",
            "No exclamation marks",
            "Spell out numbers under 10",
            "Include dateline in format: CITY, State -- (Date) --",
            "CEO quote must be enclosed in quotation marks and attributed",
            "Boilerplate paragraph must begin with 'About [Company Name]'",
        ],
    },
    {
        "voice": "confident yet accessible",
        "word_limit": 500,
        "required_sections": [
            "Headline",
            "Dateline and lead",
            "Key features (bulleted)",
            "Quote from executive",
            "Pricing and availability",
            "Company boilerplate",
            "Contact information",
        ],
        "rules": [
            "Lead paragraph must answer who, what, when, where, why",
            "Avoid jargon; define technical terms on first use",
            "Do not use superlatives (e.g., 'best', 'revolutionary')",
            "All product names in title case",
            "Boilerplate must be under 75 words",
        ],
    },
]

DEPARTMENTS = [
    "Engineering",
    "Marketing",
    "Sales",
    "Customer Support",
    "Human Resources",
    "Finance",
    "Operations",
    "Legal",
    "Product",
    "Research & Development",
    "IT Infrastructure",
    "Data Science",
]

BUDGET_REQUEST_TEMPLATES = [
    {
        "item": "Cloud infrastructure expansion",
        "justification": "Current capacity at 87% utilization; projected to hit 100% by Q3. "
        "Expansion needed to support new product launch and 40% YoY user growth.",
        "priority": "Critical",
    },
    {
        "item": "Employee training and development program",
        "justification": "Retention surveys show 62% of departing employees cite lack of "
        "growth opportunities. Industry benchmark is 2-3% of payroll on L&D.",
        "priority": "High",
    },
    {
        "item": "Office renovation and ergonomic upgrades",
        "justification": "Building lease renewal requires compliance with updated fire safety "
        "codes. Ergonomic assessment found 45% of workstations below OSHA guidelines.",
        "priority": "Medium",
    },
    {
        "item": "Marketing campaign for Q2 product launch",
        "justification": "New product entering competitive market segment. Competitor spend "
        "averages $1.2M on launch campaigns. Without comparable investment, projected "
        "market share is 3-5% vs target of 12%.",
        "priority": "High",
    },
    {
        "item": "Cybersecurity audit and remediation",
        "justification": "Last penetration test revealed 3 critical and 7 high-severity "
        "vulnerabilities. SOC 2 recertification due in 6 months. Insurance renewal "
        "contingent on remediation evidence.",
        "priority": "Critical",
    },
    {
        "item": "Customer support tooling upgrade",
        "justification": "Current ticketing system end-of-life in Q4. Average resolution time "
        "is 4.2 days vs industry benchmark of 1.8 days. CSAT score dropped 12 points YoY.",
        "priority": "High",
    },
    {
        "item": "Recruiting and talent acquisition",
        "justification": "22 open headcount across engineering and product. Current time-to-fill "
        "is 68 days. Revenue impact of unfilled roles estimated at $45K/month per role.",
        "priority": "High",
    },
    {
        "item": "Legal compliance and regulatory filing",
        "justification": "New data privacy regulations (DPDP Act) effective next quarter. "
        "Non-compliance penalties up to 2% of global revenue. Requires external counsel "
        "and internal process changes.",
        "priority": "Critical",
    },
    {
        "item": "R&D prototype lab equipment",
        "justification": "Current equipment is 7 years old; calibration failures causing 15% "
        "experiment reruns. New equipment reduces prototype cycle from 6 weeks to 3 weeks.",
        "priority": "Medium",
    },
    {
        "item": "Sustainability and ESG reporting initiative",
        "justification": "Board mandate to publish first ESG report by year-end. Requires "
        "carbon accounting software, third-party audit, and dedicated analyst. "
        "Key institutional investors have flagged ESG as condition for continued investment.",
        "priority": "Medium",
    },
    {
        "item": "Sales enablement platform",
        "justification": "Sales team relies on ad-hoc slide decks and outdated collateral. "
        "Win rate has declined from 28% to 21% over past two quarters. "
        "Platform would centralize content and provide analytics on engagement.",
        "priority": "Medium",
    },
    {
        "item": "Disaster recovery and business continuity",
        "justification": "Current RTO is 48 hours; board-mandated target is 4 hours. "
        "Last DR test revealed backup restoration failures for 3 of 8 critical systems.",
        "priority": "High",
    },
]


# ============================================================================
# 1. MEETING MINUTES
# ============================================================================

def make_meeting_minutes(rand_seed: int = 42) -> RubricDatapoint:
    """Given raw meeting transcript, produce structured minutes.

    Generates a realistic meeting transcript (~2000 words) with action items,
    decisions, and off-topic digressions. The agent must produce clean,
    structured minutes.
    """
    rng = _random.Random(rand_seed)

    topic = rng.choice(MEETING_TOPICS)
    company = rng.choice(COMPANY_NAMES)
    num_attendees = rng.randint(4, 6)
    attendees = make_names(rand_seed, num_attendees)
    facilitator = attendees[0]
    meeting_date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    # Generate 3-5 action items
    num_actions = rng.randint(3, 5)
    action_templates = [
        ("Draft the revised {} proposal", "by end of next week"),
        ("Schedule follow-up meeting with {} stakeholders", "within 5 business days"),
        ("Circulate the updated {} budget figures", "by Friday"),
        ("Prepare a competitive analysis for the {} initiative", "before the next quarterly review"),
        ("Send the {} status update to the executive team", "by Wednesday"),
        ("Compile customer feedback on the {} rollout", "within 2 weeks"),
        ("Review and finalize the {} documentation", "by end of month"),
    ]
    topic_word = topic.split()[0]
    selected_actions = rng.sample(action_templates, num_actions)
    action_items = []
    for i, (template, deadline) in enumerate(selected_actions):
        assignee = attendees[i % len(attendees)]
        action_items.append({
            "task": template.format(topic_word),
            "assignee": assignee,
            "deadline": deadline,
        })

    # Generate 2-3 decisions
    decision_templates = [
        "The team agreed to proceed with Option B for the {} strategy, contingent on board approval.",
        "Budget for {} will be increased by 15% to accommodate the expanded scope.",
        "The timeline for the {} phase will be extended by two weeks to ensure quality.",
        "All {} deliverables will require sign-off from both the project lead and the department head.",
        "The team decided to postpone the {} vendor evaluation until Q3.",
    ]
    num_decisions = rng.randint(2, 3)
    decisions = [
        rng.choice(decision_templates).format(topic_word)
        for _ in range(num_decisions)
    ]
    # Deduplicate if the same template was picked twice
    decisions = list(dict.fromkeys(decisions))[:num_decisions]

    # Build the transcript
    off_topic_segments = [
        (
            "{name}: Oh, before we move on -- did anyone see the email about the "
            "parking garage closure next week? Apparently they're resurfacing levels "
            "2 and 3.\n"
            "{name2}: Yeah, I saw that. We'll need to use the overflow lot on Pine "
            "Street. It's about a 10-minute walk.\n"
            "{name}: That's annoying. Anyway, back to the agenda..."
        ),
        (
            "{name}: Sorry, quick aside -- the coffee machine on the 4th floor is "
            "broken again. I've submitted a facilities ticket but if anyone has a "
            "contact there, that would help.\n"
            "{name2}: I'll text Maria in facilities. She fixed it last time.\n"
            "{name}: Thanks. Okay, where were we..."
        ),
        (
            "{name}: One more thing -- happy birthday to {name2} this Friday! "
            "We're doing cake in the break room at 3 PM if anyone wants to swing by.\n"
            "{name2}: Ha, thanks! You didn't have to announce it.\n"
            "{name}: Of course I did. Alright, let's get back on track."
        ),
    ]
    off_topic = rng.choice(off_topic_segments).format(
        name=attendees[-1], name2=attendees[-2]
    )

    # Compose the full transcript
    transcript_lines = []
    transcript_lines.append(f"MEETING TRANSCRIPT")
    transcript_lines.append(f"Date: {meeting_date}")
    transcript_lines.append(f"Subject: {topic}")
    transcript_lines.append(f"Company: {company}")
    transcript_lines.append(f"Attendees: {', '.join(attendees)}")
    transcript_lines.append(f"Facilitator: {facilitator}")
    transcript_lines.append("")
    transcript_lines.append(f"[Recording begins]")
    transcript_lines.append("")

    # Opening
    transcript_lines.append(
        f"{facilitator}: Good morning everyone. Thanks for joining. Let's get "
        f"started with today's agenda on {topic}. I want to make sure we cover "
        f"the key updates, make some decisions on the open items, and assign "
        f"next steps before we wrap up."
    )
    transcript_lines.append("")

    # Status updates section
    transcript_lines.append(
        f"{attendees[1]}: Sure, I can kick things off. So since our last meeting, "
        f"the team has completed the initial assessment phase for {topic_word}. "
        f"We reviewed the data from the past quarter and identified three main "
        f"areas that need attention. First, there's the resource allocation "
        f"question -- we're currently spread thin across multiple workstreams. "
        f"Second, the timeline is tighter than we originally scoped. And third, "
        f"we've received some feedback from the client advisory board that "
        f"suggests we should consider a phased rollout rather than a big-bang "
        f"launch."
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{facilitator}: That's helpful context. {attendees[2]}, can you walk us "
        f"through where things stand on the resource side?"
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{attendees[2]}: Absolutely. Right now we have four full-time people "
        f"dedicated to this, which is two fewer than what we originally requested. "
        f"The gap is mainly on the analytics side. I've been pulling in "
        f"{attendees[3]} part-time to cover, but it's not sustainable. We "
        f"either need to hire or re-prioritize some of the secondary workstreams."
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{attendees[3]}: I can confirm that. I'm currently splitting my time "
        f"between this project and the customer retention initiative, and both "
        f"are suffering. My recommendation is that we make a call on priorities "
        f"today."
    )
    transcript_lines.append("")

    # Decision discussion
    transcript_lines.append(
        f"{facilitator}: Agreed. Let's talk about that. I think we have two "
        f"options on the table. Option A: we keep the current staffing and push "
        f"the deadline out by three weeks. Option B: we bring in two contractors "
        f"to fill the gap and keep the original timeline. Thoughts?"
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{attendees[1]}: I'm in favor of Option B. The market window is "
        f"narrow and I don't want us to lose momentum. The cost of two "
        f"contractors for eight weeks is manageable within our discretionary "
        f"budget."
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{attendees[2]}: I'd second that, but I want to flag that we'll "
        f"need board approval for the additional spend. {facilitator}, can "
        f"you take that to the exec meeting on Thursday?"
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{facilitator}: Yes, I'll present it. Let's formally agree: "
        f"{decisions[0]}"
    )
    transcript_lines.append("")

    # Off-topic digression
    transcript_lines.append(off_topic)
    transcript_lines.append("")

    # Timeline discussion
    if len(attendees) > 4:
        timeline_speaker = attendees[4]
    else:
        timeline_speaker = attendees[2]
    transcript_lines.append(
        f"{timeline_speaker}: On the timeline front, I've put together a "
        f"revised Gantt chart. The critical path runs through the data "
        f"migration and the UAT phase. If we start contractors by the 15th, "
        f"we can hit the original delivery date with about a week of buffer."
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{facilitator}: Good. And {attendees[1]}, you mentioned client "
        f"feedback about a phased rollout. Can you elaborate?"
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{attendees[1]}: Right. Three of our top-ten clients expressed "
        f"concern about a full cutover. They'd prefer a pilot phase with "
        f"two or three early adopters, then general availability four weeks "
        f"later. I think that's actually a smart approach -- it reduces "
        f"our risk significantly."
    )
    transcript_lines.append("")
    if len(decisions) > 1:
        transcript_lines.append(
            f"{facilitator}: I agree. Let's make that official. {decisions[1]}"
        )
        transcript_lines.append("")
    if len(decisions) > 2:
        transcript_lines.append(
            f"{attendees[2]}: One more thing before we move to action items -- "
            f"I think we should also decide on the review process. {decisions[2]}"
        )
        transcript_lines.append("")

    # Budget tangent
    transcript_lines.append(
        f"{attendees[3]}: Quick question on budget -- do the contractor costs "
        f"come out of the {topic_word} budget line or the general operations "
        f"line?"
    )
    transcript_lines.append("")
    transcript_lines.append(
        f"{facilitator}: Good question. For now they'll come from operations, "
        f"but I'll confirm with Finance. If it needs to shift, I'll let "
        f"everyone know by end of day."
    )
    transcript_lines.append("")

    # Action items section
    transcript_lines.append(
        f"{facilitator}: Alright, let's lock down the action items. I have "
        f"the following:"
    )
    transcript_lines.append("")
    for item in action_items:
        transcript_lines.append(
            f"{facilitator}: {item['assignee']} -- {item['task']}, {item['deadline']}."
        )
    transcript_lines.append("")

    # Agreement
    for a in attendees[1:]:
        transcript_lines.append(f"{a}: Got it.")
    transcript_lines.append("")

    # Closing
    transcript_lines.append(
        f"{facilitator}: Great. Thanks everyone. Let's reconvene next week "
        f"same time. If anything comes up before then, drop it in the "
        f"Slack channel. Have a good rest of your day."
    )
    transcript_lines.append("")
    transcript_lines.append("[Recording ends]")

    transcript = "\n".join(transcript_lines)

    # Build action items string for rubric reference
    action_items_str = "; ".join(
        f"{a['assignee']}: {a['task']}" for a in action_items
    )

    problem_statement = f"""# Task: Meeting Minutes from Transcript

You have a raw meeting transcript in /testbed/transcript.txt. Read it and produce
clean, structured meeting minutes.

Your minutes must include:
- Meeting metadata (date, subject, attendees, facilitator)
- A concise summary of discussion topics (not a verbatim transcript)
- A "Decisions" section listing each decision made
- An "Action Items" section with assignee, task, and deadline for each item
- Exclude off-topic discussions, side conversations, and social banter

Format the output as a professional document suitable for distribution to
stakeholders who were not present.

Write the minutes to /testbed/minutes.txt"""

    rubric = (
        BinaryRubricCategory(
            name="all_action_items_captured",
            question=(
                f"Does the minutes document capture all {num_actions} action items "
                f"from the transcript? The action items are: {action_items_str}"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_assignees",
            question=(
                "Are the correct assignees listed for each action item? "
                "The expected assignee-to-task mapping is: "
                + "; ".join(
                    f"{a['assignee']} -> '{a['task']}'" for a in action_items
                )
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="decisions_section_present",
            question=(
                "Is there a clearly labeled 'Decisions' section (or equivalent heading) "
                "that lists the decisions made during the meeting?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="all_decisions_captured",
            question=(
                f"Are all {len(decisions)} decisions captured? The decisions are: "
                + "; ".join(decisions)
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="off_topic_excluded",
            question=(
                "Is the off-topic or social discussion (e.g., parking garage, coffee "
                "machine, birthday) excluded from the minutes?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="attendee_list_present",
            question=(
                f"Does the document list the attendees? Expected attendees: "
                f"{', '.join(attendees)}"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="date_noted",
            question=f"Is the meeting date ({meeting_date}) noted in the minutes?",
            points=1,
        ),
        RubricCategory(
            name="deadlines_included",
            description="Does each action item include a deadline or timeframe as stated in the transcript?",
            failure="No deadlines or timeframes included with any action item.",
            minor_failure="Deadlines included for fewer than half the action items.",
            minor_success="Deadlines included for most action items but one or two are missing or inaccurate.",
            success="All action items include the correct deadline or timeframe from the transcript.",
            points=2,
        ),
        RubricCategory(
            name="formatting_quality",
            description="Is the document well-formatted with clear headings, consistent structure, and professional presentation?",
            failure="No structure; reads as raw text dump or copy-paste of transcript.",
            minor_failure="Some headings present but inconsistent formatting or poor visual hierarchy.",
            minor_success="Clear section headings and mostly consistent formatting with minor issues.",
            success="Professional formatting with clear hierarchy, consistent style, and easy scannability.",
            points=3,
        ),
        RubricCategory(
            name="conciseness",
            description="Are the discussion summaries concise and distilled rather than verbose or verbatim?",
            failure="Essentially a copy of the transcript with no summarization.",
            minor_failure="Some summarization attempted but still overly verbose or includes unnecessary detail.",
            minor_success="Good summarization overall with occasional unnecessary detail.",
            success="Concise, well-distilled summaries that capture the substance without excess.",
            points=3,
        ),
        BinaryRubricCategory(
            name="facilitator_noted",
            question=f"Is {facilitator} identified as the meeting facilitator or chair?",
            points=1,
        ),
        BinaryRubricCategory(
            name="subject_noted",
            question=f"Is the meeting subject/topic ({topic}) noted in the minutes?",
            points=1,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write the meeting minutes to /testbed/minutes.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/transcript.txt": transcript,
        },
        problem_type="meeting_minutes",
    )


# ============================================================================
# 2. CUSTOMER COMPLAINT RESPONSE
# ============================================================================

def make_customer_complaint_response(rand_seed: int = 42) -> RubricDatapoint:
    """Given complaint, product specs, and policy doc, draft a response.

    The agent must read all source files and compose a professional,
    empathetic response that cites the correct policy and offers an
    appropriate remedy.
    """
    rng = _random.Random(rand_seed)

    issue = rng.choice(COMPLAINT_ISSUES)
    company = rng.choice(COMPANY_NAMES)
    customer_name = make_name(rand_seed)
    agent_name = make_name(rand_seed + 1000)
    order_number = f"ORD-{rng.randint(100000, 999999)}"
    contact_email = f"support@{company.lower().replace(' ', '').replace('.', '')}.com"
    contact_phone = f"1-800-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"

    complaint_text = (
        f"From: {customer_name}\n"
        f"To: Customer Support\n"
        f"Subject: Problem with Order #{order_number} - {issue['product']}\n"
        f"Date: January 15, 2025\n"
        f"\n"
        f"Dear Customer Support,\n"
        f"\n"
        f"I am writing to express my frustration with my recent purchase. "
        f"I ordered the {issue['product']} (SKU: {issue['sku']}) on your "
        f"website and am very disappointed.\n"
        f"\n"
        f"{issue['complaint_detail']}\n"
        f"\n"
        f"This is unacceptable for a product at this price point. I have "
        f"been a loyal customer for three years and this is the first time "
        f"I've had an issue, but it has seriously shaken my confidence in "
        f"your company.\n"
        f"\n"
        f"I expect this to be resolved promptly. Please let me know what "
        f"you can do.\n"
        f"\n"
        f"Regards,\n"
        f"{customer_name}\n"
        f"Order #: {order_number}\n"
    )

    return_policy_text = (
        f"{company} Return and Warranty Policy\n"
        f"{'=' * 50}\n"
        f"Effective Date: January 1, 2025\n"
        f"\n"
        f"1. RETURN WINDOW\n"
        f"   - Standard returns accepted within {issue['return_window']} days of delivery.\n"
        f"   - Items must be in original packaging with all tags/accessories.\n"
        f"   - Refund processed within 5-7 business days of receiving the return.\n"
        f"\n"
        f"2. DEFECTIVE PRODUCTS\n"
        f"   - Products with manufacturing defects may be exchanged or refunded\n"
        f"     at any time within the warranty period.\n"
        f"   - Proof of purchase required (order number or receipt).\n"
        f"   - {company} covers return shipping for defective items.\n"
        f"\n"
        f"3. WRONG ITEM RECEIVED\n"
        f"   - If you received the wrong item or size, contact us within 14 days.\n"
        f"   - We will send the correct item with free expedited shipping.\n"
        f"   - A prepaid return label will be provided for the incorrect item.\n"
        f"\n"
        f"4. LATE DELIVERY\n"
        f"   - If a guaranteed delivery date is missed, customers are eligible for:\n"
        f"     a) Full refund of shipping charges\n"
        f"     b) $25 store credit for the inconvenience\n"
        f"   - Claims must be filed within 30 days of the original delivery date.\n"
        f"\n"
        f"5. MISSING PARTS\n"
        f"   - If your product is missing components listed in the manual,\n"
        f"     contact us within 60 days of delivery.\n"
        f"   - Replacement parts will be shipped free of charge via expedited\n"
        f"     shipping.\n"
        f"\n"
        f"6. CONTACT INFORMATION\n"
        f"   - Email: {contact_email}\n"
        f"   - Phone: {contact_phone}\n"
        f"   - Hours: Monday-Friday 8:00 AM - 8:00 PM EST\n"
        f"   - Chat: Available 24/7 on our website\n"
    )

    problem_statement = f"""# Task: Customer Complaint Response

You have three files to review:

1. /testbed/complaint.txt - A customer complaint email
2. /testbed/product_specs.txt - Specifications for the product in question
3. /testbed/policy/return_policy.txt - The company's return and warranty policy

Read all three files, then write a professional customer service response that:
- Acknowledges the customer's frustration and the specific issue
- References the correct product by name and SKU
- Cites the applicable policy section and offers the appropriate remedy
- Provides clear next steps for the customer
- Includes company contact information
- Maintains a professional and empathetic tone throughout

Write the response to /testbed/response.txt"""

    rubric = (
        BinaryRubricCategory(
            name="complaint_acknowledged",
            question=(
                "Does the response acknowledge the customer's specific complaint "
                f"about the {issue['type'].replace('_', ' ')} issue?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_product_referenced",
            question=(
                f"Does the response reference the correct product name "
                f"({issue['product']}) and/or SKU ({issue['sku']})?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_policy_cited",
            question=(
                "Does the response cite or reference the applicable section of the "
                f"return/warranty policy (the '{issue['type'].replace('_', ' ').upper()}' "
                f"or related section)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="appropriate_remedy_offered",
            question=(
                f"Does the response offer an appropriate remedy consistent with the "
                f"policy? For this issue type ('{issue['type'].replace('_', ' ')}'), "
                f"an appropriate remedy would be: {issue['remedy']}."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="contact_info_included",
            question=(
                f"Does the response include contact information such as email "
                f"({contact_email}) or phone ({contact_phone})?"
            ),
            points=1,
        ),
        RubricCategory(
            name="professional_tone",
            description="Is the response written in a professional, customer-service-appropriate tone?",
            failure="Unprofessional, rude, sarcastic, or blames the customer.",
            minor_failure="Mostly professional but with overly casual language or missing salutation/closing.",
            minor_success="Professional tone with proper salutation and closing; minor stylistic issues.",
            success="Polished, warm, professional tone with appropriate salutation, closing, and consistent voice throughout.",
            points=2,
        ),
        BinaryRubricCategory(
            name="customer_name_used",
            question=f"Does the response address the customer by name ({customer_name})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="order_number_referenced",
            question=f"Does the response reference the order number ({order_number})?",
            points=1,
        ),
        RubricCategory(
            name="empathy_quality",
            description="How well does the response demonstrate genuine empathy and understanding of the customer's frustration?",
            failure="No empathy shown; response is robotic or dismissive.",
            minor_failure="Generic empathy statement (e.g., 'We're sorry') without addressing the specific frustration.",
            minor_success="Shows understanding of the specific issue and acknowledges the inconvenience.",
            success="Genuinely empathetic; validates the customer's experience, acknowledges the impact, and expresses sincere concern.",
            points=3,
        ),
        RubricCategory(
            name="solution_clarity",
            description="How clear and actionable are the next steps provided to the customer?",
            failure="No clear next steps; customer would not know what to do after reading.",
            minor_failure="Next steps mentioned vaguely but lack specificity (no timeline, no process).",
            minor_success="Clear next steps with most details, minor gaps in process or timeline.",
            success="Crystal clear action plan with specific steps, timeline, and what the customer should expect.",
            points=3,
        ),
        BinaryRubricCategory(
            name="no_fabricated_policy",
            question=(
                "Does the response avoid fabricating policy details that are not in "
                "the provided return_policy.txt? (It should only cite what is actually "
                "in the policy document.)"
            ),
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write the response to /testbed/response.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/complaint.txt": complaint_text,
            "/testbed/product_specs.txt": issue["product_specs"],
            "/testbed/policy/return_policy.txt": return_policy_text,
        },
        problem_type="complaint_response",
    )


# ============================================================================
# 3. COMPETITIVE COMPARISON
# ============================================================================

def make_competitive_comparison(rand_seed: int = 42) -> RubricDatapoint:
    """Given 3 product spec sheets, write a comparison document.

    Generates three software products with comparable features where each
    product excels at different things. The agent must produce an accurate,
    analytical comparison.
    """
    rng = _random.Random(rand_seed)

    product_set = rng.choice(SOFTWARE_PRODUCT_SETS)
    category = product_set["category"]
    products = product_set["products"]
    best_at = product_set["best_at"]

    # Build spec sheets
    spec_files = {}
    product_labels = ["product_a", "product_b", "product_c"]
    feature_keys = list(products[0].keys())
    feature_keys.remove("name")

    for i, (label, prod) in enumerate(zip(product_labels, products)):
        lines = [
            f"{'=' * 50}",
            f"PRODUCT SPECIFICATION SHEET",
            f"{'=' * 50}",
            f"",
            f"Product Name: {prod['name']}",
            f"Category: {category}",
            f"",
            f"--- Features ---",
        ]
        for key in feature_keys:
            display_key = key.replace("_", " ").title()
            lines.append(f"  {display_key}: {prod[key]}")
        lines.append("")
        lines.append(f"--- End of Specification ---")
        spec_files[f"/testbed/specs/{label}.txt"] = "\n".join(lines)

    product_names = [p["name"] for p in products]

    # Identify the cheapest product (for rubric)
    # We'll pick two specific feature comparisons for binary checks
    # Feature 1: storage -- find the winner
    storage_values = {}
    for p in products:
        s = p["storage"]
        if s == "Unlimited":
            storage_values[p["name"]] = float("inf")
        else:
            # Extract numeric part
            numeric = "".join(c for c in s if c.isdigit() or c == ".")
            storage_values[p["name"]] = float(numeric) if numeric else 0
    storage_winner = max(storage_values, key=storage_values.get)

    # Feature 2: uptime SLA -- find the winner
    uptime_values = {}
    for p in products:
        numeric = "".join(c for c in p["uptime_sla"] if c.isdigit() or c == ".")
        uptime_values[p["name"]] = float(numeric) if numeric else 0
    uptime_winner = max(uptime_values, key=uptime_values.get)

    problem_statement = f"""# Task: Competitive Product Comparison

You have three product specification sheets for {category} solutions:

- /testbed/specs/product_a.txt
- /testbed/specs/product_b.txt
- /testbed/specs/product_c.txt

Read all three spec sheets and write a comprehensive comparison document that:
- Compares all three products across their key features
- Identifies which product is best for different use cases
- Accurately reflects the specifications (do not invent or misstate data)
- Includes a summary table or structured comparison
- Provides a recommendation section explaining when to choose each product

Write the comparison to /testbed/comparison.txt"""

    rubric = (
        BinaryRubricCategory(
            name="all_products_compared",
            question=(
                f"Does the comparison document discuss all three products "
                f"({', '.join(product_names)})?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_storage_winner",
            question=(
                f"Does the document correctly identify {storage_winner} as having "
                f"the most storage (or unlimited storage, if applicable)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_uptime_winner",
            question=(
                f"Does the document correctly identify {uptime_winner} as having "
                f"the best (highest) uptime SLA ({uptime_values[uptime_winner]}%)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="pricing_accurate",
            question=(
                "Are the pricing figures stated in the comparison accurate to what "
                "is in the spec sheets? Check that each product's price matches: "
                + ", ".join(f"{p['name']}: {p['price']}" for p in products)
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="no_fabricated_specs",
            question=(
                "Does the document avoid fabricating specifications that are not "
                "in the provided spec sheets? (No invented features, metrics, or "
                "numbers that are not present in the source files.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="all_features_covered",
            question=(
                f"Does the comparison cover at least {len(feature_keys) - 1} of the "
                f"{len(feature_keys)} feature dimensions present in the spec sheets "
                f"({', '.join(k.replace('_', ' ') for k in feature_keys)})?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="use_case_recommendations",
            question=(
                "Does the document include recommendations for different use cases "
                "or buyer profiles (e.g., small team vs enterprise, budget-conscious "
                "vs feature-rich)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="structured_comparison",
            question=(
                "Does the document include a structured comparison (such as a table, "
                "side-by-side layout, or clearly organized feature-by-feature "
                "comparison)?"
            ),
            points=1,
        ),
        RubricCategory(
            name="analysis_depth",
            description="How insightful is the analysis beyond just restating specs?",
            failure="No analysis; merely lists specs without any comparison or insight.",
            minor_failure="Some comparison but largely restates specs without drawing conclusions.",
            minor_success="Good analysis with meaningful comparisons; identifies tradeoffs between products.",
            success="Excellent analysis that highlights tradeoffs, identifies best value propositions, and provides nuanced reasoning.",
            points=3,
        ),
        RubricCategory(
            name="recommendation_quality",
            description="How useful and well-reasoned are the recommendations?",
            failure="No recommendations or recommendations contradicted by the data.",
            minor_failure="Vague recommendations that don't connect to specific user needs.",
            minor_success="Reasonable recommendations tied to user scenarios but missing some nuance.",
            success="Well-reasoned recommendations for specific buyer profiles with clear justification from the data.",
            points=3,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write the comparison to /testbed/comparison.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=spec_files,
        problem_type="competitive_comparison",
    )


# ============================================================================
# 4. PRESS RELEASE
# ============================================================================

def make_press_release(rand_seed: int = 42) -> RubricDatapoint:
    """Given product info, company facts, and style guide, draft a press release.

    The agent must synthesize information from three documents and produce
    a press release following the style guide's constraints.
    """
    rng = _random.Random(rand_seed)

    product = rng.choice(PRESS_RELEASE_PRODUCTS)
    company = rng.choice(COMPANY_NAMES)
    ceo_name = make_name(rand_seed + 500)
    style = rng.choice(STYLE_GUIDE_TEMPLATES)
    founded_year = rng.randint(2005, 2018)
    employee_count = rng.choice([150, 250, 500, 800, 1200, 2500])
    hq_city = rng.choice([
        "San Francisco, CA", "Austin, TX", "Boston, MA", "Seattle, WA",
        "Denver, CO", "New York, NY", "Chicago, IL", "Portland, OR",
    ])
    media_contact_name = make_name(rand_seed + 600)
    media_email = f"press@{company.lower().replace(' ', '').replace('.', '')}.com"

    product_info_text = (
        f"PRODUCT INFORMATION - CONFIDENTIAL UNTIL {product['launch_date'].upper()}\n"
        f"{'=' * 60}\n"
        f"\n"
        f"Product Name: {product['product_name']}\n"
        f"Tagline: {product['tagline']}\n"
        f"Launch Date: {product['launch_date']}\n"
        f"\n"
        f"Key Features:\n"
    )
    for feat in product["key_features"]:
        product_info_text += f"  - {feat}\n"
    product_info_text += (
        f"\n"
        f"Pricing: {product['price']}\n"
        f"Availability: {product['availability']}\n"
        f"\n"
        f"Target Audience: Mid-market and enterprise companies\n"
        f"Competitive Positioning: First-to-market with integrated advanced capabilities "
        f"in this category\n"
    )

    company_facts_text = (
        f"COMPANY FACT SHEET\n"
        f"{'=' * 40}\n"
        f"\n"
        f"Company Name: {company}\n"
        f"Founded: {founded_year}\n"
        f"Headquarters: {hq_city}\n"
        f"CEO: {ceo_name}\n"
        f"Employees: {employee_count}+\n"
        f"Mission: To empower organizations with innovative technology solutions\n"
        f"\n"
        f"Recent Milestones:\n"
        f"  - Named to Inc. 5000 list of fastest-growing companies (2024)\n"
        f"  - Closed Series C funding of $45M (2023)\n"
        f"  - Expanded to 3 international offices\n"
        f"  - 500+ enterprise customers across 30 countries\n"
        f"\n"
        f"Boilerplate (use verbatim):\n"
        f"About {company}: {company} is a leading provider of technology "
        f"solutions for modern enterprises. Founded in {founded_year} and "
        f"headquartered in {hq_city}, the company serves over 500 customers "
        f"worldwide. For more information, visit www.{company.lower().replace(' ', '').replace('.', '')}.com.\n"
        f"\n"
        f"Media Contact:\n"
        f"  {media_contact_name}\n"
        f"  Director of Communications\n"
        f"  {media_email}\n"
        f"  (555) 123-4567\n"
    )

    style_guide_text = (
        f"PRESS RELEASE STYLE GUIDE\n"
        f"{'=' * 40}\n"
        f"\n"
        f"Voice: {style['voice']}\n"
        f"Word Limit: {style['word_limit']} words maximum\n"
        f"\n"
        f"Required Sections (in order):\n"
    )
    for i, section in enumerate(style["required_sections"], 1):
        style_guide_text += f"  {i}. {section}\n"
    style_guide_text += f"\nStyle Rules:\n"
    for rule in style["rules"]:
        style_guide_text += f"  - {rule}\n"
    style_guide_text += (
        f"\n"
        f"Additional Notes:\n"
        f"  - Press release must include a placeholder for CEO quote: use the "
        f"    CEO's actual name and a quote that speaks to the product's value\n"
        f"  - Include the company boilerplate verbatim from the fact sheet\n"
        f"  - End with '###' to indicate end of press release\n"
    )

    problem_statement = f"""# Task: Press Release Drafting

You have three documents to work from:

1. /testbed/docs/product_info.txt - Product details and launch information
2. /testbed/docs/company_facts.txt - Company background and boilerplate
3. /testbed/docs/style_guide.txt - Press release formatting requirements

Read all three files and draft a press release that follows the style guide's
requirements, accurately represents the product, and uses the company
information correctly.

Key requirements:
- Follow the section order specified in the style guide
- Stay within the word limit
- Include the CEO quote (attributed to the correct CEO)
- Include the boilerplate verbatim
- Include media contact information

Write the press release to /testbed/press_release.txt"""

    rubric = (
        BinaryRubricCategory(
            name="correct_product_name",
            question=(
                f"Does the press release use the correct product name "
                f"({product['product_name']})?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_launch_date",
            question=(
                f"Does the press release include the correct launch date "
                f"({product['launch_date']})?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="ceo_quote_present",
            question=(
                f"Does the press release include a quote attributed to the CEO "
                f"({ceo_name})? The quote should be in quotation marks."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="boilerplate_included",
            question=(
                f"Does the press release include the company boilerplate paragraph "
                f"starting with 'About {company}'?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="within_word_limit",
            question=(
                f"Is the press release within the {style['word_limit']}-word limit "
                f"specified in the style guide? (Allow a 10% margin.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="contact_info_present",
            question=(
                f"Does the press release include media contact information "
                f"({media_contact_name} and/or {media_email})?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="pricing_included",
            question=(
                f"Does the press release mention the pricing information "
                f"({product['price']})?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="key_features_mentioned",
            question=(
                "Does the press release mention at least 2 of the key features "
                "listed in the product info document?"
            ),
            points=2,
        ),
        RubricCategory(
            name="required_sections_present",
            description=f"Does the press release include the required sections from the style guide ({', '.join(style['required_sections'][:4])}, ...)?",
            failure=f"Fewer than 3 of the {len(style['required_sections'])} required sections are present.",
            minor_failure=f"About half of the {len(style['required_sections'])} required sections are present.",
            minor_success=f"Most required sections present (at least {len(style['required_sections']) - 2} of {len(style['required_sections'])}) with minor omissions.",
            success=f"All {len(style['required_sections'])} required sections from the style guide are present and in the correct order.",
            points=2,
        ),
        RubricCategory(
            name="newsworthiness_framing",
            description="Does the press release frame the announcement in a newsworthy way?",
            failure="Reads like a product brochure or advertisement, not a press release.",
            minor_failure="Some press release structure but fails to convey why this matters now.",
            minor_success="Adequate framing as news with some context on market significance.",
            success="Strong newsworthy angle; clearly communicates the 'why now' and market impact.",
            points=3,
        ),
        RubricCategory(
            name="professional_tone",
            description="Does the press release maintain a professional, press-appropriate tone?",
            failure="Unprofessional, heavily promotional, or uses excessive superlatives.",
            minor_failure="Mostly professional but with some overly promotional language or inconsistent tone.",
            minor_success="Professional throughout with minor tone issues.",
            success="Polished, authoritative press tone consistent with the style guide voice requirements.",
            points=3,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write the press release to /testbed/press_release.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/docs/product_info.txt": product_info_text,
            "/testbed/docs/company_facts.txt": company_facts_text,
            "/testbed/docs/style_guide.txt": style_guide_text,
        },
        problem_type="press_release",
    )


# ============================================================================
# 5. LITERATURE SYNTHESIS (static)
# ============================================================================

def make_literature_synthesis() -> RubricDatapoint:
    """Given 4 paper abstracts and a research question, write a synthesis.

    Static problem (no seed) using realistic abstracts about the effects
    of remote work on productivity.
    """

    research_question = (
        "What does the current empirical evidence tell us about the impact "
        "of remote and hybrid work arrangements on employee productivity, "
        "and what are the key moderating factors?"
    )

    paper1_abstract = (
        "Title: Remote Work and Productivity: A Longitudinal Study of "
        "Knowledge Workers\n"
        "Authors: Chen, M., Rodriguez, A., & Patel, S. (2024)\n"
        "Journal: Journal of Organizational Behavior, 45(2), 112-131.\n"
        "\n"
        "Abstract:\n"
        "This longitudinal study tracked 2,847 knowledge workers across "
        "14 technology companies over 18 months following the transition "
        "to remote work. Using objective output measures (code commits, "
        "documents produced, tickets resolved) and manager performance "
        "ratings, we found that fully remote workers experienced an "
        "initial productivity decline of 8-12% in the first three months, "
        "followed by a recovery to baseline by month six. By month 12, "
        "remote workers showed a 4% productivity advantage over their "
        "pre-remote baseline. However, this effect was strongly moderated "
        "by home office quality (beta = 0.31, p < 0.001) and prior remote "
        "work experience (beta = 0.24, p < 0.01). Workers in shared "
        "living spaces with no dedicated office showed persistent "
        "productivity deficits of 6-9% throughout the study period. "
        "Manager ratings were systematically lower for remote workers "
        "despite equal or higher objective output, suggesting proximity "
        "bias in subjective evaluations."
    )

    paper2_abstract = (
        "Title: The Hybrid Advantage: Comparing Work Arrangements in "
        "Financial Services\n"
        "Authors: Thompson, K. & Nakamura, Y. (2023)\n"
        "Journal: Academy of Management Journal, 66(4), 987-1015.\n"
        "\n"
        "Abstract:\n"
        "We conducted a quasi-experimental study with 5,200 employees "
        "at three major financial institutions that implemented different "
        "return-to-office policies in 2022. Firm A mandated full in-office "
        "(5 days), Firm B adopted hybrid (3 days office / 2 days remote), "
        "and Firm C allowed fully flexible arrangements. Using difference-"
        "in-differences analysis with propensity score matching, we found "
        "hybrid workers at Firm B demonstrated the highest productivity "
        "(7% above Firm A and 3% above Firm C on revenue-per-employee "
        "metrics). Critically, Firm B also showed 33% lower voluntary "
        "turnover and 18% higher employee satisfaction scores. The "
        "productivity advantage of hybrid work was concentrated in "
        "collaborative roles (project managers, analysts) rather than "
        "independent contributor roles (traders, underwriters), where "
        "no significant differences emerged. We attribute this to the "
        "'structured flexibility' that hybrid arrangements provide: "
        "dedicated collaboration days plus focused deep-work days."
    )

    paper3_abstract = (
        "Title: Communication Patterns and Innovation Under Remote Work: "
        "Evidence from Patent Data\n"
        "Authors: Bergstrom, L., Okonkwo, C., & Singh, R. (2024)\n"
        "Journal: Research Policy, 53(1), 104892.\n"
        "\n"
        "Abstract:\n"
        "While most studies focus on individual productivity, we "
        "examined the impact of remote work on team-level innovation "
        "using patent filings and internal invention disclosures from "
        "12 R&D-intensive firms (N = 8,400 employees, 2019-2023). We "
        "found that remote work reduced cross-team communication by "
        "25% (measured via email, Slack, and meeting data) and that "
        "this reduction was associated with a 17% decline in novel "
        "patent filings -- patents citing prior art outside the team's "
        "usual domain. However, within-team communication actually "
        "increased by 12%, and teams showed a 9% increase in "
        "incremental patents (extensions of existing work). Notably, "
        "the innovation decline was not uniform: teams that adopted "
        "structured 'virtual collider' sessions -- scheduled cross-team "
        "brainstorming events -- showed no significant decline in novel "
        "patents compared to their pre-remote baseline. These findings "
        "suggest that remote work's negative impact on innovation is "
        "mediated by informal cross-boundary interactions, which can "
        "be partially offset by deliberate organizational design."
    )

    paper4_abstract = (
        "Title: Remote Work, Mental Health, and Sustainable Productivity: "
        "A Meta-Analysis\n"
        "Authors: Kim, J., Alvarez, D., & Osei, F. (2025)\n"
        "Journal: Psychological Bulletin, 151(1), 45-78.\n"
        "\n"
        "Abstract:\n"
        "This meta-analysis synthesized 94 studies (total N = 127,000 "
        "participants) published between 2020 and 2024 on remote work "
        "and employee outcomes. Our overall effect size for remote work "
        "on productivity was d = 0.02 (95% CI: -0.04 to 0.08), "
        "indicating no practically significant overall effect. However, "
        "substantial heterogeneity (I-squared = 78%) revealed important "
        "moderators. Remote work was associated with improved mental "
        "health outcomes (d = 0.21 for reduced burnout, d = 0.18 for "
        "work-life balance), which in turn predicted sustained long-term "
        "productivity (r = 0.34). The relationship between remote work "
        "and productivity was curvilinear: 2-3 days per week remote "
        "yielded the best outcomes, while 5 days remote showed slight "
        "negative effects. Methodological quality was a significant "
        "moderator: studies using objective productivity measures found "
        "smaller effects than those relying on self-reports (d = -0.03 "
        "vs d = 0.15), suggesting self-report bias inflates apparent "
        "benefits. We identify manager support, task autonomy, and "
        "technology infrastructure as the three strongest moderating "
        "factors."
    )

    problem_statement = """# Task: Literature Synthesis

You have four research paper abstracts and a research question:

- /testbed/papers/paper1_abstract.txt
- /testbed/papers/paper2_abstract.txt
- /testbed/papers/paper3_abstract.txt
- /testbed/papers/paper4_abstract.txt
- /testbed/research_question.txt

Read all files and write a synthesis (not just a summary of each paper) that:
- Answers the research question by integrating findings across all four papers
- Attributes specific findings to the correct papers/authors
- Identifies common themes and points of agreement
- Notes any contradictions or tensions between the studies
- Identifies gaps in the current research
- Mentions methodological limitations
- Stays under 500 words

The synthesis should integrate findings thematically rather than summarizing
each paper sequentially.

Write the synthesis to /testbed/synthesis.txt"""

    rubric = (
        BinaryRubricCategory(
            name="all_papers_cited",
            question=(
                "Does the synthesis cite or reference all four papers (Chen et al., "
                "Thompson & Nakamura, Bergstrom et al., Kim et al.)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_findings_attributed",
            question=(
                "Are the key findings correctly attributed to their papers? For "
                "example: Chen et al. found 4% productivity gain at 12 months; "
                "Thompson & Nakamura found hybrid workers 7% more productive; "
                "Bergstrom et al. found 17% decline in novel patents; Kim et al. "
                "found d=0.02 overall effect size."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="research_gap_identified",
            question=(
                "Does the synthesis identify at least one research gap or area "
                "where more evidence is needed?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="methodological_limitations",
            question=(
                "Does the synthesis mention at least one methodological limitation "
                "across the studies (e.g., self-report bias, proximity bias, "
                "industry-specific samples, or measurement challenges)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="under_500_words",
            question="Is the synthesis under 500 words? (Allow a 10% margin, so under 550.)",
            points=1,
        ),
        BinaryRubricCategory(
            name="answers_research_question",
            question=(
                "Does the synthesis directly address the research question about "
                "the impact of remote/hybrid work on productivity and key "
                "moderating factors?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="thematic_not_sequential",
            question=(
                "Is the synthesis organized thematically (e.g., by topic like "
                "productivity, innovation, moderators) rather than sequentially "
                "summarizing Paper 1, then Paper 2, then Paper 3, then Paper 4?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_hybrid_consensus",
            question=(
                "Does the synthesis identify the emerging consensus across papers "
                "that hybrid (2-3 days remote) arrangements yield the best "
                "productivity outcomes?"
            ),
            points=1,
        ),
        RubricCategory(
            name="synthesis_quality",
            description="Does the writing genuinely synthesize rather than just summarize each paper in turn?",
            failure="Simply summarizes each paper sequentially with no integration.",
            minor_failure="Attempts integration but mostly reads as sequential summaries with transitional phrases.",
            minor_success="Good thematic organization with genuine cross-paper comparisons, minor gaps.",
            success="Excellent synthesis that weaves findings together, draws new insights from the combination, and presents a coherent narrative.",
            points=3,
        ),
        RubricCategory(
            name="critical_analysis",
            description="Does the synthesis demonstrate critical thinking about the evidence?",
            failure="No critical engagement; takes all findings at face value.",
            minor_failure="Minimal critical analysis; notes one limitation without exploring implications.",
            minor_success="Thoughtful engagement with evidence quality and notes tensions between findings.",
            success="Strong critical analysis that evaluates evidence quality, identifies biases, weighs conflicting findings, and qualifies conclusions appropriately.",
            points=3,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write the synthesis to /testbed/synthesis.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/papers/paper1_abstract.txt": paper1_abstract,
            "/testbed/papers/paper2_abstract.txt": paper2_abstract,
            "/testbed/papers/paper3_abstract.txt": paper3_abstract,
            "/testbed/papers/paper4_abstract.txt": paper4_abstract,
            "/testbed/research_question.txt": research_question,
        },
        problem_type="literature_synthesis",
    )


# ============================================================================
# 6. BUDGET ALLOCATION
# ============================================================================

def make_budget_allocation(rand_seed: int = 42) -> RubricDatapoint:
    """Given department requests, costs, and budget constraints, write a proposal.

    Generates 5-8 department requests totaling 130-150% of the available
    budget, forcing the agent to make tradeoff decisions and justify them.
    """
    rng = _random.Random(rand_seed)

    company = rng.choice(COMPANY_NAMES)
    fiscal_year = rng.choice(["FY2025", "FY2026"])

    # Pick 5-8 budget requests
    num_requests = rng.randint(5, 8)
    selected_requests = rng.sample(BUDGET_REQUEST_TEMPLATES, num_requests)

    # Assign departments to requests
    departments = rng.sample(DEPARTMENTS, num_requests)

    # Generate costs that total 130-150% of budget
    base_budget = rng.choice([500_000, 750_000, 1_000_000, 1_500_000, 2_000_000])

    # Target total spend: 130-150% of budget
    target_ratio = rng.uniform(1.3, 1.5)
    target_total = base_budget * target_ratio

    # Distribute costs roughly proportionally with some variance
    raw_costs = [rng.uniform(0.5, 2.0) for _ in range(num_requests)]
    cost_sum = sum(raw_costs)
    costs = [round(c / cost_sum * target_total / 1000) * 1000 for c in raw_costs]
    # Ensure no cost is zero
    costs = [max(c, 25_000) for c in costs]
    total_requested = sum(costs)

    # Build department_requests.csv
    csv_header = "Department,Request Item,Priority,Justification"
    csv_rows = []
    for i in range(num_requests):
        dept = departments[i]
        req = selected_requests[i]
        # Escape commas in justification for CSV
        justification = req["justification"].replace('"', '""')
        csv_rows.append(
            f'{dept},{req["item"]},{req["priority"]},"{justification}"'
        )
    department_requests_csv = csv_header + "\n" + "\n".join(csv_rows) + "\n"

    # Build cost_estimates.csv
    cost_csv_header = "Department,Request Item,Estimated Cost ($),Cost Confidence"
    cost_csv_rows = []
    for i in range(num_requests):
        dept = departments[i]
        req = selected_requests[i]
        confidence = rng.choice(["High", "Medium", "Low"])
        cost_csv_rows.append(
            f'{dept},{req["item"]},{costs[i]},{confidence}'
        )
    cost_estimates_csv = cost_csv_header + "\n" + "\n".join(cost_csv_rows) + "\n"

    # Build budget constraints text
    budget_constraints_text = (
        f"{company} -- {fiscal_year} Budget Allocation Guidelines\n"
        f"{'=' * 60}\n"
        f"\n"
        f"Total Available Budget: ${base_budget:,}\n"
        f"\n"
        f"CONSTRAINTS:\n"
        f"1. Total allocations must not exceed ${base_budget:,}.\n"
        f"2. All 'Critical' priority items must receive at least 50% of their\n"
        f"   requested amount.\n"
        f"3. No single department may receive more than 40% of the total budget.\n"
        f"4. A minimum of $25,000 must be allocated to each department that\n"
        f"   submitted a request (even if reduced from original ask).\n"
        f"5. Any request reduced by more than 30% requires a written justification\n"
        f"   in the proposal.\n"
        f"\n"
        f"DECISION CRITERIA (in order of priority):\n"
        f"  a) Regulatory and compliance obligations\n"
        f"  b) Revenue protection and growth\n"
        f"  c) Employee retention and satisfaction\n"
        f"  d) Operational efficiency\n"
        f"  e) Strategic initiatives\n"
        f"\n"
        f"The proposal should include:\n"
        f"  - A summary table of allocations vs. requests\n"
        f"  - Justification for any significant reductions\n"
        f"  - Total budget utilization\n"
        f"  - Risk assessment for underfunded items\n"
    )

    # Count critical items for rubric
    critical_items = [
        (departments[i], selected_requests[i])
        for i in range(num_requests)
        if selected_requests[i]["priority"] == "Critical"
    ]
    critical_str = ", ".join(
        f"{dept}: {req['item']}" for dept, req in critical_items
    )

    problem_statement = f"""# Task: Budget Allocation Proposal

You have three files containing budget request information:

1. /testbed/data/department_requests.csv - Department requests with priorities and justifications
2. /testbed/data/cost_estimates.csv - Cost estimates for each request
3. /testbed/docs/budget_constraints.txt - Budget constraints and decision criteria

The total requested amount (${total_requested:,}) exceeds the available budget
(${base_budget:,}), so you must make tradeoff decisions.

Read all files and write a budget allocation proposal that:
- Allocates the available ${base_budget:,} budget across the requests
- Respects the constraints in the budget guidelines
- Justifies any significant reductions from requested amounts
- Includes a summary table showing requested vs. allocated amounts
- Assesses risks of underfunding any items
- Shows that the total allocation does not exceed the budget

Write the proposal to /testbed/budget_proposal.txt"""

    rubric = (
        BinaryRubricCategory(
            name="within_total_budget",
            question=(
                f"Do the proposed allocations sum to ${base_budget:,} or less? "
                f"(Check the numbers in the proposal add up correctly.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="all_departments_addressed",
            question=(
                f"Does the proposal address all {num_requests} departments that "
                f"submitted requests ({', '.join(departments)})?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="totals_add_up",
            question=(
                "Do the individual allocations in the proposal sum to the stated "
                "total? (Check basic arithmetic consistency.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="critical_priorities_respected",
            question=(
                f"Are all 'Critical' priority items allocated at least 50% of their "
                f"requested amount as required by the constraints? Critical items: "
                f"{critical_str}"
            ),
            points=2,
        ),
        RubricCategory(
            name="justification_for_cuts",
            description="Does the proposal include written justification for requests that were significantly reduced (by more than 30%)?",
            failure="No justifications provided for any reductions.",
            minor_failure="Justifications provided for some reduced items but missing for others, or justifications are perfunctory.",
            minor_success="Justifications provided for most significantly reduced items; reasoning is adequate.",
            success="Clear, well-reasoned justification provided for every item reduced by more than 30%, referencing the decision criteria.",
            points=2,
        ),
        BinaryRubricCategory(
            name="summary_table_present",
            question=(
                "Does the proposal include a summary table (or structured list) "
                "showing requested amount vs. allocated amount for each department?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="minimum_allocation_met",
            question=(
                "Does every department receive at least $25,000 as required by "
                "the constraints?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="no_department_exceeds_40pct",
            question=(
                f"Does no single department receive more than 40% of the total "
                f"budget (${int(base_budget * 0.4):,})?"
            ),
            points=1,
        ),
        RubricCategory(
            name="strategic_reasoning",
            description="How well-reasoned are the allocation decisions and tradeoffs?",
            failure="No reasoning; allocations appear arbitrary with no justification.",
            minor_failure="Some reasoning but decisions don't clearly connect to the prioritization criteria.",
            minor_success="Good reasoning that references the priority framework; most tradeoffs are explained.",
            success="Excellent strategic reasoning that systematically applies the decision criteria, explains tradeoffs, and demonstrates clear prioritization logic.",
            points=3,
        ),
        RubricCategory(
            name="presentation_quality",
            description="How professional and clear is the overall proposal document?",
            failure="Poorly organized; hard to follow or missing essential sections.",
            minor_failure="Basic structure present but inconsistent formatting or unclear sections.",
            minor_success="Well-structured document with clear sections and mostly professional presentation.",
            success="Executive-ready document with clear formatting, logical flow, professional tone, and easy-to-scan structure.",
            points=3,
        ),
        BinaryRubricCategory(
            name="risk_assessment_included",
            question=(
                "Does the proposal include a risk assessment or discussion of "
                "consequences for items that received reduced funding?"
            ),
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write the budget proposal to /testbed/budget_proposal.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/department_requests.csv": department_requests_csv,
            "/testbed/data/cost_estimates.csv": cost_estimates_csv,
            "/testbed/docs/budget_constraints.txt": budget_constraints_text,
        },
        problem_type="budget_allocation",
    )
