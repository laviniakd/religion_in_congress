## print unique addresses to csv file to input to census geocoder api
import pandas as pd
import re
from tqdm import tqdm
#from geopy.geocoders import Nominatim
from geocodio import GeocodioClient
import json

with open('projects/sermons-ir/geocodio_api_key.txt') as f:
    API_KEY = f.read().strip()
    
client = GeocodioClient('537b5463643656343555645335536367bb36765')

OUTPUT_PATH = '/home/laviniad/projects/religion_in_congress/unparsed_addresses_data.json'
NON_PARSED_PATH = '/home/laviniad/projects/religion_in_congress/unparsedaddresses.txt'
DEBUG = True
pattern = r'(\d+\s+[^,]+),\s*([^,]+(?:\s[^,]+)?),\s*([^ ]+)\s+(\d{5})'
NON_AMER_CONST = '*Province/Other' # the sermoncentral shibboleth for non-US addresses

with open(NON_PARSED_PATH) as f:
    non_parsed_addresses_list = [l.strip() for l in f.readlines()]

MAX_NUM_REQ = 2000
end_idx = max(len(non_parsed_addresses_list), MAX_NUM_REQ)

address_list = client.geocode(non_parsed_addresses_list[:end_idx], fields=["cd", "census2016", "acs-demographics", "acs-economics", "acs-families", "acs-social"])

with open(OUTPUT_PATH, 'w') as f:
    json.dump(address_list, f)
