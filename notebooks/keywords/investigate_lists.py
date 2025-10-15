list_one = ["Abed\-nego, abideth, Abram, adul\-ter\-ers, Ahab, AMEN, Ana\-nias, apos\-tle, apos\-tles, Aren, ark, Baal, Bap\-tism, Be\-liev\-ers, be\-lieveth, Beth\-phage, Bible, Boaz, bride\-groom, Ca\-iaphas, cen\-tu\-ri\-on, Cephas, Christ, Chron, Colos\-sians, con\-ceit\-ed, Cor, Corinthi\-ans, cov\-etous\-ness, De\-ity, De\-liv\-er\-er, dis\-ci\-ple, dis\-ci\-ples, dis\-obe\-di\-ent, dis\-sen\-sions, Di\-vin\-i\-ty, doeth, dwelleth, Eph\-esians, Eph\-esus, faith, FAITH, Gala\-tia, Gala\-tians, Galilee, GEN\-E\-SIS, Gen\-tiles, God, GOD, god\-less\-ness, god\-li\-ness, God\-li\-ness, god\-ly, God\-ly, Habakkuk, Haman, Ha\-ran, hast, He\-brews, Herod, Him, Him\-self, ho\-li\-ness, Idol, idol\-a\-try, Im\-mor\-tal, im\-per\-ish\-able, in\-iq\-ui\-ties, in\-iq\-ui\-ty, Isa, Is\-raelites, Jabez, Jeph\-thah, Je\-sus, Jezebel, Joab, Ju\-dah, Ju\-das, KJV, knoweth, le\-gal\-ism, Lep\-rosy, Levite, liveth, Lystra, Ma\-gi, meek\-ness, MER\-CY, Me\-shach, Mes\-si\-ah, NASB, Neb\-uchad\-nez\-zar, Ne\-he\-mi\-ah, Obe\-di\-ence, Olives, OUR, para\-ble, Para\-ble, para\-bles, par\-tak\-ers, Pen\-te\-cost, Pharaoh, Phar\-isees, Philip\-pi, Philip\-pi\-an, Philip\-pi\-ans, Philis\-tine, Philistines, Pi\-late, pray, prayed, prayer, pray\-ing, prophets, Prophets, Ra\-hab, Re\-deemer, re\-pen\-tance, Re\-pen\-tance, res\-ur\-rec\-tion, Rev\-e\-la\-tion, righ\-teous\-ness, Righ\-teous\-ness, Rom, sack\-cloth, saith, sal\-va\-tion, sanc\-ti\-fi\-ca\-tion, San\-hedrin, Sa\-tan, Sav\-ior, Saviour, scoffers, seeth, Ser\-mon, Shadrach, sin\-ful, sin\-ful\-ness, sin\-ner, sin\-ners, Sodom, sow\-er, Sow\-er, spake, Swin\-doll, taber\-na\-cle, Thes\-sa\-lo\-ni\-ans, un\-be\-lief, un\-be\-liev\-er, un\-be\-liev\-ers, un\-be\-liev\-ing, un\-de\-filed, un\-fruit\-ful, un\-god\-li\-ness, un\-righ\-teous, un\-righ\-teous\-ness, Ver\-i\-ly, whoso\-ev\-er, wicked\-ness, william, wine\-skins, Your\-selves, Za\-c\-cha\-eus, Zechari\-ah, Zedeki\-ah"]
list_one = [e.replace('\\-', '') for e in list_one[0].split(", ")]

list_two = []
with open("/home/laviniad/projects/religion_in_congress/src/keyword_list_construction/full_list_new.txt") as f:
	words = [e.strip() for e in f.readlines()]
	list_two = words

print("Paper list:")
print(len(list_one))

print("List in dir:")
print(len(list_two))

print("Difference, 1 - 2:")
print(set(list_one) - set(list_two))

print("Difference, 2 - 1:")
print(set(list_two) - set(list_one))
