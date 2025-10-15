echo "Creating a list of word frequencies"
python /home/laviniad/projects/religion_in_congress/src/keyword_list_construction/get_token_frequencies.py --prep_tokens
echo "Using log-odds scores to create a word list"
python /home/laviniad/projects/religion_in_congress/src/keyword_list_construction/log_odds.py --prep_tokens

echo "Done creating keyword list"
echo "Converting to .txt"
python /home/laviniad/projects/religion_in_congress/src/keyword_list_construction/convert_keyword_json_to_txt.py

