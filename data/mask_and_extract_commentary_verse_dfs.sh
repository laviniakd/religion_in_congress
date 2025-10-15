echo "------------ Now masking quotes... ------------"
python /home/laviniad/projects/religion_in_congress/data/mask_quotes.py
echo "------------ Now separating into train, dev, and test... "
python /home/laviniad/projects/religion_in_congress/data/separate_train_test_dev.py
echo "------------ Now converting train to commentary-verse pairs... ------------"
python /home/laviniad/projects/religion_in_congress/src/convert_sermons_to_commentaryversepairs.py
