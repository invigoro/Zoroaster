"""Whether a name says which country it belongs to: "Ukraine's armed forces", "the Ethiopian government".

The checks treat a country's government or armed forces as the country itself, which the user's rules allow to
do anything (2026-10-04). The model alone called the Rapid Support Forces, Sudan's paramilitary at war with its
army, a country's forces, and the UN Security Council too. So a government or a force counts as a country's only
if its name names the country, as the user's own examples did: "Ukraine's armed forces", "Saudi Arabia's air
force". A nationality counts for its country. The list misses some, which only makes the checks stricter.
"""

from __future__ import annotations

import re

COUNTRIES = (
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia", "Botswana", "Brazil", "Britain", "Brunei", "Bulgaria",
    "Burkina Faso", "Burma", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Central African Republic",
    "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia",
    "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador", "Egypt",
    "El Salvador", "England", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland",
    "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
    "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Korea", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saint Lucia", "Samoa", "San Marino", "Saudi Arabia", "Scotland", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Türkiye", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican", "Venezuela", "Vietnam",
    "Wales", "Yemen", "Zambia", "Zimbabwe",
)
# Nationalities that don't start like their country, and a multi-word country's own first word.
NATIONALITIES = ("american", "belgian", "british", "burmese", "chinese", "cypriot", "czech", "danish", "dutch",
                 "emirati", "english", "filipino", "finnish", "french", "greek", "icelandic", "irish", "italian",
                 "kazakh", "kyrgyz", "lao", "lebanese", "malagasy", "monegasque", "mozambican", "polish", "saudi",
                 "scottish", "slovak", "spanish", "swedish", "swiss", "tajik", "thai", "turkish", "turkmen", "uzbek",
                 "welsh")
ABBREVIATIONS = re.compile(r"(?<!\w)(?:U\.S\.|US|USA|UK|UAE|DPRK|PRC|ROK)(?!\w)")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _prefix(country: str) -> str:
    """How much of a one-word country's name its nationality shares: "ukrain" for Ukrainian, "sudan" for Sudanese.
    All but the last letter, at least five: with two fewer, "liberation" started like Liberia."""
    lower = country.lower()
    return lower if len(lower) <= 5 else lower[:max(5, len(lower) - 1)]


SINGLE = tuple(_prefix(c) for c in COUNTRIES if " " not in c)
MULTI = tuple(c.lower() for c in COUNTRIES if " " in c)


def names_a_country(name: str) -> bool:
    """Whether `name` names a country, or a nationality: "Israel Defense Forces", "the Sudanese government",
    "U.S. Army". Not "Rapid Support Forces" or "United Nations Security Council"."""
    if ABBREVIATIONS.search(name):
        return True
    lower = name.lower()
    if any(re.search(rf"(?<!\w){re.escape(country)}", lower) for country in MULTI):
        return True
    return any(word in NATIONALITIES or word.startswith(SINGLE) for word in WORD.findall(lower))
