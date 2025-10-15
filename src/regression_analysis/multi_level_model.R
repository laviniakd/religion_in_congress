library(jsonlite)
library(lme4)
library(stargazer)
library(dplyr)

print("** Loading scdata **")
full_path <- "/data/laviniad/congress_errata/classed_cdf_readable.json" # no text, for size's sake
raw_df <- jsonlite::stream_in(textConnection(readLines(full_path)),verbose=F)

raw_df <- raw_df %>% 
    filter(!is.na(bio_id)) %>% 
    filter(!is.na(lexical)) %>% 
    filter(!is.na(is_republican)) %>% 
    filter(!is.na(is_in_senate)) %>% 
    filter(!is.na(year)) %>% 
    filter(!is.na(state))

print("** Loaded data **")

df <- as.data.frame(raw_df)

print("Number of rows in df:")
print(nrow(df))

lr_df <- raw_df %>% 
    filter(!is.na(avg_lr_prob))
lr_df <- as.data.frame(lr_df)

print("Number of rows in lr_df:")
print(nrow(lr_df))

print("NOW REGRESSING ON CLASSIFIER OUTPUT")
print("** Iterating through various models: proportion marked religious **")

print("Single variables")
party_model <- lm(lr_label_prop_religious ~ is_republican, data = df)
year_model <- lm(lr_label_prop_religious ~ year, data = df)

nore_model <- lm(lr_label_prop_religious ~ is_republican + is_in_senate + is_republican * year, data = df)
multi_model <- lm(lr_label_prop_religious ~ is_republican + is_in_senate * is_republican + is_in_senate + is_republican * year + (1 | year), data = df)
stargazer(party_model, year_model, nore_model, multi_model, type = "html", align=TRUE, single.row=TRUE)

print("** Iterating through various models: average classifier probability **")

print("Single variables")
party_model <- lm(avg_lr_prob ~ is_republican, data = df)

year_model <- lm(avg_lr_prob ~ year, data = df)


print("No random effects")
nore_model <- lm(avg_lr_prob ~ is_republican + is_in_senate + is_republican * year, data = df)
multi_model <- lm(avg_lr_prob ~ is_republican + is_in_senate * is_republican + is_in_senate + is_republican * year + (1 | year), data = df)
stargazer(party_model, year_model, nore_model, multi_model, type = "html", align=TRUE, single.row=TRUE)


print("NOW REGRESSING ON KEYWORD-BASED MEASURES")

print("** Iterating through various models **")

party_model <- lm(lexical ~ is_republican, data = df)
year_model <- lm(lexical ~ year, data = df)

nore_model <- lm(lexical ~ is_republican + is_in_senate + is_republican * year, data = df)

multi_model <- lm(lexical ~ is_republican + is_in_senate * is_republican + is_in_senate + is_republican * year + (1 | year), data = df)
stargazer(party_model, year_model, nore_model, multi_model, type = "html", align=TRUE, single.row=TRUE)

print("118th Congress-specific...")
data_118 <- df %>% dplyr::filter(congress_num == 118)
house_df <- data_118 %>% dplyr::filter(is_in_senate == 0)

print("Only county religious adherence, House only: ")
relig_adherent_model <- lm(lexical ~ perc_adherents, data = house_df)
multi_model <- lm(lexical ~ is_republican + is_in_senate + is_republican * is_in_senate + is_republican * perc_adherents, data = house_df)
stargazer(relig_adherent_model, multi_model, type = "html", align=TRUE, single.row=TRUE)

print("** Redoing with speaker-level grouping **")
speaker_df <- df %>% group_by(bio_id) %>% summarise(lexical = mean(lexical), is_republican = first(is_republican), congress_num = first(congress_num), is_in_senate = first(is_in_senate), year = first(year), religion = first(religion), perc_adherents = first(perc_adherents), state = first(state), state_perc_white = first(state_perc_white), state_perc_black = first(state_perc_black))

speaker_party_model <- lm(lexical ~ is_republican, data = speaker_df)

speaker_chamber_model <- lm(lexical ~ is_in_senate, data = speaker_df)

speaker_year_model <- lm(lexical ~ year, data = speaker_df)

nore_model <- lm(lexical ~ is_republican + is_in_senate + is_republican * year, data = speaker_df)

multi_model <- lm(lexical ~ is_republican + is_republican * is_in_senate + is_in_senate + is_republican * year + (1 | year), data = speaker_df)
stargazer(speaker_party_model, speaker_chamber_model, speaker_year_model, nore_model, multi_model, type = "html", align=TRUE, single.row=TRUE)

print("118th Congress-specific...")
print("speaker_df has the following columns: ")
print(names(speaker_df))
data_118 <- dplyr::filter(speaker_df, congress_num == 118) # line is not working for some reason
print("data_118 has the following columns: ")
print(names(data_118))
data_118$religion <- as.factor(data_118$religion)

print("Only county religious adherence: ")
model_percadherent <- lm(lexical ~ perc_adherents, data = data_118)
model_onlyrelig <- lm(lexical ~ religion, data = data_118)
multi_model <- lm(lexical ~ is_republican + is_in_senate + + is_republican * is_in_senate + is_republican * perc_adherents + religion + is_republican * religion, data = data_118)

stargazer(model_percadherent, model_onlyrelig, multi_model, type = "html", align=TRUE, single.row=TRUE)


