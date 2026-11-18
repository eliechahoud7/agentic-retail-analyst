"""Readable display names for database label values (shared by the report stage and the UI).

CATEGORY_NAMES covers every value of category_reporting in retail_intelligence (74 values,
checked by tests/test_ui.py against the live database). Names are formatted, not translated:
the words are the database's own; only spacing, capitalisation and separators are added.
"_2" categories are separate categories in the database, so their suffix is kept.
"""
from __future__ import annotations

CATEGORY_NAMES = {
    "agro_industry_and_commerce": "Agro Industry & Commerce",
    "air_conditioning": "Air Conditioning",
    "art": "Art",
    "arts_and_craftmanship": "Arts & Craftmanship",
    "audio": "Audio",
    "auto": "Auto",
    "baby": "Baby",
    "bed_bath_table": "Bed, Bath & Table",
    "books_general_interest": "Books: General Interest",
    "books_imported": "Books: Imported",
    "books_technical": "Books: Technical",
    "cds_dvds_musicals": "CDs, DVDs & Musicals",
    "christmas_supplies": "Christmas Supplies",
    "cine_photo": "Cine & Photo",
    "computers": "Computers",
    "computers_accessories": "Computers & Accessories",
    "consoles_games": "Consoles & Games",
    "construction_tools_construction": "Construction Tools: Construction",
    "construction_tools_garden": "Construction Tools: Garden",
    "construction_tools_lights": "Construction Tools: Lights",
    "construction_tools_safety": "Construction Tools: Safety",
    "construction_tools_tools": "Construction Tools: Tools",
    "cool_stuff": "Cool Stuff",
    "diapers_and_hygiene": "Diapers & Hygiene",
    "drinks": "Drinks",
    "dvds_blu_ray": "DVDs & Blu-ray",
    "electronics": "Electronics",
    "fashion_bags_accessories": "Fashion: Bags & Accessories",
    "fashion_childrens_clothes": "Fashion: Children's Clothes",
    "fashion_female_clothing": "Fashion: Female Clothing",
    "fashion_male_clothing": "Fashion: Male Clothing",
    "fashion_shoes": "Fashion: Shoes",
    "fashion_sport": "Fashion: Sport",
    "fashion_underwear_beach": "Fashion: Underwear & Beach",
    "fixed_telephony": "Fixed Telephony",
    "flowers": "Flowers",
    "food": "Food",
    "food_drink": "Food & Drink",
    "furniture_bedroom": "Furniture: Bedroom",
    "furniture_decor": "Furniture & Decor",
    "furniture_living_room": "Furniture: Living Room",
    "furniture_mattress_and_upholstery": "Furniture: Mattress & Upholstery",
    "garden_tools": "Garden Tools",
    "health_beauty": "Health & Beauty",
    "home_appliances": "Home Appliances",
    "home_appliances_2": "Home Appliances 2",
    "home_comfort": "Home Comfort",
    "home_comfort_2": "Home Comfort 2",
    "home_construction": "Home Construction",
    "housewares": "Housewares",
    "industry_commerce_and_business": "Industry, Commerce & Business",
    "kitchen_dining_laundry_garden_furniture": "Kitchen, Dining, Laundry & Garden Furniture",
    "la_cuisine": "La Cuisine",
    "luggage_accessories": "Luggage & Accessories",
    "market_place": "Market Place",
    "music": "Music",
    "musical_instruments": "Musical Instruments",
    "office_furniture": "Office Furniture",
    "party_supplies": "Party Supplies",
    "pc_gamer": "PC Gamer",
    "perfumery": "Perfumery",
    "pet_shop": "Pet Shop",
    "portable_kitchen_and_food_preparers": "Portable Kitchen & Food Preparers",
    "security_and_services": "Security & Services",
    "signaling_and_security": "Signaling & Security",
    "small_appliances": "Small Appliances",
    "small_appliances_home_oven_and_coffee": "Small Appliances: Home, Oven & Coffee",
    "sports_leisure": "Sports & Leisure",
    "stationery": "Stationery",
    "tablets_printing_image": "Tablets, Printing & Image",
    "telephony": "Telephony",
    "toys": "Toys",
    "uncategorized": "Uncategorized",
    "watches_gifts": "Watches & Gifts",
}


OTHER_NAMES = {  # catalog grouping values that need wording, not just capitalisation
    "on_time": "On-time deliveries",
    "late": "Late deliveries",
}


def display_label(value) -> str:
    """Readable form of a label value: known categories by mapping; other snake_case values
    ('credit_card') by spacing and capitalisation; codes such as 'AL' or '2018-01' unchanged."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text_ = str(value)
    if text_ in CATEGORY_NAMES:
        return CATEGORY_NAMES[text_]
    if text_ in OTHER_NAMES:
        return OTHER_NAMES[text_]
    if text_ == text_.lower() and any(ch.isalpha() for ch in text_):
        return text_.replace("_", " ").strip().title()
    return text_
