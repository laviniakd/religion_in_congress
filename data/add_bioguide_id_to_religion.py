import csv
import os
import re
import json

directory = "/home/laviniad/projects/religion_in_congress/data/pol_data/"
csv_files = [directory + f for f in ['housereligion.csv', 'senatereligion.csv']]
bioguide_file = "/data/laviniad/congress_bioguides.jsonlist"
bioguide_to_religion_path = '/data/laviniad/congress_errata/bioguide_to_religion.json'
citation_regex = r'\[\d*\]' # for normalizing out wikipedia citations -- appear as e.g. "[6]"

# unfortunately, people have lots of different ways of recording their name in the bioguide vis a vis their name on wikipedia
def normalize(name):
    if name == 'Jaime Herrera Beutler':
        return name.lower()
    elif name == 'Joe Manchin III':
        return 'joe manchin'
    elif name == 'Donald S. Beyer Jr.':
        return 'donald beyer'
    elif name == 'Van Taylor':
        return 'nicholas taylor'
    elif name == 'Patrick J. Toomey':
        return 'patrick toomey'
    elif name == "Patrick T. McHenry":
        return 'patrick mchenry'
    elif name == 'William R. Timmons IV':
        return 'william timmons'
    elif name == "A. Donald McEachin":
        return 'aston mceachin'
    elif name == "Tom Malinowski":
        return name.lower()
    elif name == "Deb Haaland":
        return 'debra haaland'
    elif name == "Ben Sasse":
        return 'benjamin sasse'
    elif name == "Frank Pallone Jr.":
        return 'frank pallone'
    elif name == "Ted Deutch":
        return 'theodore deutch'
    elif name == "G.K. Butterfield":
        return 'george butterfield'
    elif name == "Fred Upton":
        return 'frederick upton'
    elif name == "Patrick J. Leahy":
        return 'patrick leahy'
    elif name == "Earl L. \"Buddy\" Carter":
        return 'buddy carter'
    elif name == "Cindy Axne":
        return 'cynthia axne'
    elif name == "Trey Hollingsworth":
        return 'joseph hollingsworth'
    elif name == "C.A. Dutch Ruppersberger":
        return 'c. a. ruppersberger'
    elif name == "Sanford D. Bishop Jr.":
        return 'sanford bishop'
    elif name == "Val B. Demings":
        return 'valdez demings'
    elif name == "Jackie Speier":
        return 'karen speier'
    elif name == "Bill Pascrell Jr.":
        return 'william pascrell'
    elif name == "Al Lawson":
        return 'alfred lawson'
    elif name == "Kai Kahele":
        return 'kaiali\'i kahele'
    elif name == "Emanuel Cleaver II":
        return 'emanuel cleaver'
    elif name == "Brendan F. Boyle":
        return 'brendan boyle'
    elif name == 'Jim Hagedorn':
        return 'jim hagedorn'
    elif name == "Mike Doyle":
        return 'michael doyle'

    # 118th exceptions
    if name == 'Lisa Blunt Rochester':
        return "lisa blunt rochester" # special case, name is recorded oddly
    elif name == "Don Bacon":
        return 'donald bacon'
    elif name == "Jim Jordan":
        return 'jim jordan'
    elif name == "Mike Turner":
        return 'michael turner'
    elif name == "Morgan Griffith":
        return 'h. griffith'
    elif name == "Derrick Van Orden":
        return 'derrick van orden'
    elif name == "Tom McClintock":
        return 'tom mcclintock'
    elif name == "Vern Buchanan":
        return 'vernon buchanan'
    elif name == "Hal Rogers":
        return 'harold rogers'
    elif name == "Donald M. Payne Jr.":
        return 'donald payne'
    elif name == "Bonnie Watson Coleman":
        return 'bonnie watson coleman'
    elif name == "Shontel Brown":
        return 'shontel  brown'
    elif name == "Chip Roy":
        return 'charles roy'
    elif name == "Marc Veasey":
        return 'marc veasey'
    elif name == "Gwen Moore":
        return 'gwendolynne moore'
    elif name == "Greg Steube":
        return 'w.  steube'
    elif name == "Dutch Ruppersberger":
        return 'c. a. ruppersberger'
    elif name == "Tom Cole":
        return 'tom cole'
    elif name == "Dan Bishop":
        return 'dan bishop'
    elif name == "Rick Crawford":
        return 'rick crawford'
    elif name == "Rick W. Allen":
        return 'rick allen'
    elif name == "Tim Walberg":
        return 'tim walberg'
    elif name == "Ken Calvert":
        return 'ken calvert'
    elif name == "Andy Barr":
        return 'garland barr'
    elif name == "Thomas Kean Jr.":
        return 'thomas kean'
    elif name == "Monica De La Cruz":
        return 'monica de la cruz'
    elif name == "Beth Van Duyne":
        return 'beth van duyne'
    elif name == "Bobby Scott":
        return 'robert scott'
    elif name == "Don Beyer":
        return 'donald beyer'
    elif name == "Jack Bergman":
        return 'john bergman'
    elif name == "Drew Ferguson":
        return 'anderson ferguson'
    elif name == "Cathy McMorris Rodgers":
        return 'cathy mcmorris rodgers'
    elif name == "Sheila Jackson Lee":
        return 'sheila jackson lee'
    elif name == "Greg Stanton":
        return 'greg stanton'
    elif name == "Mike Thompson":
        return 'charles thompson'
    elif name == "Jimmy Panetta":
        return 'james panetta'
    elif name == "Jim Costa":
        return 'jim costa'
    elif name == "Lou Correa":
        return 'jose correa'
    elif name == "Joe Courtney":
        return 'joe courtney'
    elif name == "Bill Keating":
        return 'william keating'
    elif name == "Pete Sessions":
        return 'pete sessions'
    elif name == "Dan Kildee":
        return 'dan kildee'
    elif name == "Mike Flood":
        return 'mike  flood'
    elif name == "Raja Krishnamoorthi":
        return 's. krishnamoorthi'
    elif name == 'Chrissy Houlahan':
        return 'christina houlahan'
    elif name == "Debbie Wasserman Schultz":
        return name.lower()
    elif name == "Greg Landsman":
        return 'greg landsman'
    elif name == "Andy Biggs":
        return 'andrew biggs'
    elif name == "Mike Simpson":
        return 'michael simpson'
    elif name == "Ted Cruz":
        return 'rafael cruz'
    elif name == "Deb Fisher":
        return 'deborah fisher'
    elif name == "Chris Coons":
        return 'christopher coons'
    elif name == "Chuck Grassley":
        return 'charles grassley'
    elif name == "Mitt Romney":
        return 'willard romney'
    elif name == "J. D. Vance":
        return 'james vance'
    elif name == "Chuck Schumer":
        return 'charles schumer'
    elif name == "Mike Crapo":
        return 'michael crapo'
    elif name == "Jacky Rosen":
        return 'jacklyn rosen'
    elif name == "Jon Ossoff":
        return 'thomas ossoff'
    elif name == "Bernie Sanders":
        return 'bernard sanders'
    elif name == "Ron Wyden":
        return 'ronald wyden'
    elif name == "Ben Cardin":
        return 'benjamin cardin'
    elif name == "Thom Tillis":
        return 'thomas tillis'
    elif name == "Alex Padilla":
        return 'alejandro padilla'
    elif name == "Jack Reed":
        return 'john reed'
    elif name == "Mike Rounds":
        return 'marion rounds'
    elif name == "Mike Braun":
        return 'michael braun'
    elif name == "Bob Casey":
        return 'robert casey'
    elif name == "Dick Durbin":
        return 'richard durbin'
    elif name == "Catherine Cortez Masto":
        return name.lower()
    elif name == "Joe Manchin":
        return name.lower()
    elif name == "Ed Markey":
        return 'edward markey'
    elif name == "Bob Menendez":
        return 'robert menendez'
    elif name == "Pete Ricketts":
        return 'john ricketts'
    elif name == "Deb Fischer":
        return 'debra fischer'
    elif name == "Josh Hawley":
        return 'joshua hawley'
    elif name == "Mitch McConnell":
        return 'addison mcconnell'
    elif name == "Tommy Tuberville":
        return 'thomas tuberville'
    elif name == "Maggie Hassan":
        return 'margaret hassan'
    elif name == "Tim Scott":
        return name.lower()
    elif name == "Chris Van Hollen":
        return name.lower().replace("chris ", "christopher ")
    elif name == "Ted Budd":
        return 'theodore budd'
    elif name == "Bill Hagerty":
        return 'william hagerty'
    elif name == "Chris Murphy":
        return 'christopher murphy'
    elif name == "Tom Cotton":
        return name.lower()
    elif name == "Debbie Stabenow":
        return 'deborah stabenow'
    elif name == "Andy Harris":
        return 'andy harris'
    elif name == 'Bill Pascrell':
        return 'william pascrell'
    elif name == 'Marcy Kaptur':
        return 'marcia kaptur'
    elif name == 'Bob Latta':
        return 'robert latta'
    elif name == 'Brendan Boyle':
        return 'brendan boyle'
    elif name == 'Dan Meuser':
        return 'dan meuser'
    elif name == 'Ben Cline':
        return 'benjamin cline'
    elif name == 'Jen Kiggans':
        return 'jennifer kiggans'
    elif name == 'Greg Casar':
        return 'greg casar'
    elif name == 'Jerry Nadler':
        return 'jerrold nadler'
    elif name == 'Gabe Amo':
        return name.lower()
    elif name == 'Jeff Van Drew':
        return 'jefferson van drew'
    elif name == 'Chris Smith':
        return 'christopher smith'
    elif name == "Dina Titus":
        return 'alice titus'
    elif name == "Jan Schakowsky":
        return 'janice schakowsky'
    elif name == "Steve Cohen":
        return 'stephen cohen'
    elif name == "Mike Gallagher":
        return 'michael gallagher'
    elif name == "Mike Lawler":
        return 'michael lawler'
    elif name == 'Matt Cartwright':
        return 'matt cartwright'
    elif name == 'Patrick McHenry':
        return 'patrick mchenry'
    elif name == "Pat Ryan":
        return 'patrick ryan'
    elif name == "Gerry Connolly":
        return 'gerald connolly'
    elif name == "Teresa Leger Fernandez":
        return 'teresa leger fernandez'

    if len(name.split()) > 2: # has middle name
        name = name.split()[0] + " " + name.split()[-1]

    return name.lower().replace("gabe ", "gabriel ").replace("pete ", "peter ").replace("angie ", "angela ").replace("tim ", "timothy ").replace("rob ", "robert ").replace("lizzie ", "elizabeth ").replace("rick ", "richard ").replace("greg ", "gregory ").replace("dan ", "daniel ").replace("dusty ", "dustin ").replace("sam ", "samuel ").replace("matt ", "matthew ").replace("tom ", "thomas ").replace("joe ", "joseph ").replace("nick ", "nicholas ").replace("ken ", "kenneth ").replace("marc ", "marcus ").replace("jim ", "james ").replace("russ ", "russell ").replace("ú", "u").replace("é", "e").replace("á", "a").replace("í", "i")

bioguide_to_religion = {}

print("Looking at 118th...")
with open(bioguide_file, "r") as bioguidejson:
    id_to_name = {}
    for l in bioguidejson.readlines():
        obj = json.loads(l.strip())
        id_to_name[obj["usCongressBioId"]] = obj["unaccentedGivenName"] + " " + obj["unaccentedFamilyName"] 
        # assume will work for 99% of cases
        
    # is ordered from oldest entries to most recent, so id is most recent associated with name in question -- works for e.g. donald payne jr.
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}

    for file in csv_files:
        with open(file, "r", encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)[1:] # drop first -- header

        for row in rows:
            #print(row)
            if "Senator" in row.keys():
                name = row['Senator']
            else:
                name = row['Representative']
            name = normalize(name)

            if name not in name_to_id:
                print("row: ", row)
                print((f"Name {name} not found in bioguide file").upper())
            else:
                id = name_to_id[name]
                row["bioguide"] = id
                bioguide_to_religion[id] = re.sub(citation_regex, '', row["Religion"], count=10)
                # max 10 citations

        # Write the updated rows back to the CSV file
        with open(file, "w", newline="") as csv_file:
            fieldnames = reader.fieldnames + ["bioguide"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# now add in 117
path_117 = "/home/laviniad/projects/religion_in_congress/data/religious_affliations_117.csv"

print("Now 117th...")
with open(path_117, "r", encoding='utf-8-sig') as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)[1:] # drop first -- header
    for row in rows:
        #print(row)
        name = row['Name']
        name = normalize(name)

        if name not in name_to_id:
            print("row: ", row)
            print((f"Name {name} not found in bioguide file").upper())
        else:
            id = name_to_id[name]
            row["bioguide"] = id
            bioguide_to_religion[id] = re.sub(citation_regex, '', row["Religion"], count=10)


with open(bioguide_to_religion_path, "w") as f:
    json.dump(bioguide_to_religion, f, indent=4, sort_keys=True)
    print(f"Dumped to {f.name}")
