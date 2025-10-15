## print unique addresses to csv file to input to census geocoder api
import pandas as pd
import re
from tqdm import tqdm

OUTPUT_PATH = '/home/laviniad/projects/religion_in_congress/addresses.csv'
SERMON_DF_PATH = '/data/laviniad/sermons-ir/sermoncentral/with_columns.csv'
NON_PARSED_PATH = '/home/laviniad/projects/religion_in_congress/unparsedaddresses.txt'
DEBUG = True
pattern = r'(\d+\s+[^,]+),\s*([^,]+(?:\s[^,]+)?),\s*([^ ]+)\s+(\d{5})'
NON_AMER_CONST = '*Province/Other' # the sermoncentral shibboleth for non-US addresses

df = pd.read_csv(SERMON_DF_PATH)
new_df = []

# inspired by https://pe.usps.com/text/pub28/28apc_002.htm
STREET_CONSTS = ['Avenue', 'Ave', 'Ave.', 'Street', 'St', 'St.', 'Road', 'Rd', 'Rd.', 'Route', 'Rte.', 'Rte', 'Rt', 'Rt.',
                'Blvd', 'Blvd.', 'Boulevard', 'Dr', 'Dr.', 'Drive', 'Way', 'Bridge', 'Parkway', 'Pkwy', 'Pkwy.', 'Ridgeway']

# parse address from string into [address, city, state, zip]
# example input: 954 N. Old Stage Road Mount Shasta, California 96097
# needs [Unique ID, Street address, City, State, ZIP]

def extract_address_components(address_str):
    if NON_AMER_CONST in address_str:
        return 0
    
    # Regular expression pattern for P.O. Box addresses
    po_box_pattern = r'P\.?O\.? Box (\d+),?\s*([^,]+),\s*([^ ]+)\s+(\d{5})'

    po_box_match = re.match(po_box_pattern, address_str)
    if po_box_match:
        po_box_number = po_box_match.group(1).strip()
        city = po_box_match.group(2).strip()
        state = po_box_match.group(3).strip()
        zip_code = po_box_match.group(4).strip()
        return [''.join(["P.O. Box " + po_box_number, city]), None, state, zip_code]

    # Regular expression pattern for standard addresses
    standard_pattern = r'(\d+\s+[^,]+),?\s*([^,]+(?:\s[^,]+)?),\s*([^ ]+)\s+(\d{5})'

    match = re.match(standard_pattern, address_str)
    if match:
        address = match.group(1).strip()
        city = match.group(2).strip()
        state = match.group(3).strip()
        zip_code = match.group(4).strip()
        return [''.join([address, city]), None, state, zip_code]

    return None


unique_addresses = set(df['churchAddress'])
non_amer = []
non_parsed = []

for loc_string in tqdm(unique_addresses):
    #res = parse_address(loc_string)
    #if res is not None:
    res = extract_address_components(loc_string)
    if res == 0:
        non_amer.append(loc_string)
    elif res is not None:
        new_df.append({'address': res[0],
                      'city': res[1],
                      'state': res[2],
                      'zip': res[3]})
    else:
        non_parsed.append(loc_string)

# for debuggin
with open(NON_PARSED_PATH, 'w') as f:
    f.writelines([l + '\n' for l in non_parsed])
    
print(f'Number of non-American locations: {len(non_amer)}')
print(f'Number of non-parsed but possibly American locations: {len(non_parsed)}')
pd.DataFrame(new_df).to_csv(OUTPUT_PATH)
